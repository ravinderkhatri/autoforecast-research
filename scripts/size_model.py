#!/usr/bin/env python3
"""Parameter count and training-memory budget for the target architecture.

Every number the accompanying analysis quotes is produced here, so the arithmetic is
checkable and the assumptions are editable in one place.

Usage:
    python3 scripts/size_model.py                 # base case
    python3 scripts/size_model.py --skus 40000    # override any cardinality
    python3 scripts/size_model.py --no-patch      # cost of NOT patching the daily axis
"""
from __future__ import annotations

import argparse

MB = 1024**2
GB = 1024**3


# ---------------------------------------------------------------- embedding sizing
def emb_size(n: int, max_size: int = 100) -> int:
    """pytorch-forecasting's rule, kept for continuity with the incumbent model."""
    return min(round(1.6 * n**0.56), max_size) if n > 2 else 1


# ---------------------------------------------------------------- layer cost models
def sa_layer_params(d: int) -> int:
    """Pre-LN self-attention encoder layer: 4 projections + FFN(4d) + 2 LayerNorms."""
    mha = 4 * d * d + 4 * d
    ffn = d * (4 * d) + (4 * d) + (4 * d) * d + d     # two linears with biases
    ln = 2 * (2 * d)
    return mha + ffn + ln


def xa_layer_params(d: int) -> int:
    """Decoder layer: self-attention + cross-attention + FFN + 3 LayerNorms."""
    return 2 * (4 * d * d + 4 * d) + (d * 4 * d + 4 * d + 4 * d * d + d) + 3 * (2 * d)


def act_floats(n_groups: int, tokens_per_group: int, d: int, heads: int) -> int:
    """Activations retained for backward, in floats, for one sample, one layer.

    Six T*d tensors (LN out, Q, K, V, attention out, projection out), one T*4d FFN
    hidden, and the attention probability matrix. Attention is block-diagonal over
    n_groups independent groups, so the quadratic term pays per group, not globally.
    """
    T = n_groups * tokens_per_group
    return 10 * T * d + heads * n_groups * tokens_per_group**2


# ---------------------------------------------------------------- the model
def size(cfg) -> dict:
    d, h = cfg.d_model, cfg.heads
    out: dict = {}

    # ---- embeddings -------------------------------------------------------
    # The decisive choice. A JOINT embedding indexed by (branch, SKU) is what the
    # move to branch-level grain tempts you into; a FACTORISED pair is what makes
    # the model trainable at all. Both are computed so the gap is explicit.
    e_branch = emb_size(cfg.branches)
    joint_pairs = cfg.branches * cfg.skus
    out["emb_joint_series"] = joint_pairs * emb_size(joint_pairs)
    out["emb_branch"] = cfg.branches * e_branch

    # How SKU identity is represented. A free per-SKU table is the default and the
    # worst option: it is 70% of the model, it has no row for a new SKU, and it needs
    # re-indexing every time the assortment churns.
    rep = cfg.sku_rep
    if rep == "free":                       # 20,000 x 100 lookup table
        e_sku = emb_size(cfg.skus)
        out["emb_sku"] = cfg.skus * e_sku
    elif rep == "attributes":               # composed from brand/category/pack only
        e_sku = 0
        out["emb_sku"] = 0
    elif rep == "residual":                 # attributes + a small shrunk per-SKU residual
        e_sku = cfg.sku_residual_dim
        out["emb_sku_residual"] = cfg.skus * e_sku
    elif rep == "hashed":                   # k hash buckets, unbounded vocabulary
        e_sku = cfg.d_model
        out["emb_sku_hashed"] = cfg.sku_buckets * e_sku
    elif rep == "text":                     # frozen sentence embedding of sku_desc
        e_sku = cfg.d_model
        out["proj_sku_text"] = cfg.text_dim * cfg.d_model + cfg.d_model
    elif rep == "text+residual":            # the recommendation
        e_sku = cfg.d_model + cfg.sku_residual_dim
        out["proj_sku_text"] = cfg.text_dim * cfg.d_model + cfg.d_model
        out["emb_sku_residual"] = cfg.skus * cfg.sku_residual_dim
    else:
        raise SystemExit(f"unknown --sku-rep {rep}")

    static_cards = {
        "brand": cfg.brands,
        "category": cfg.categories,
        "pack_size": cfg.pack_sizes,
        "branch_type": cfg.branch_types,
        "branch_group": cfg.branch_groups,
    }
    static_dims = {"branch": e_branch, "sku": e_sku}
    for name, card in static_cards.items():
        dim = emb_size(card)
        static_dims[name] = dim
        out[f"emb_{name}"] = card * dim

    cal_cards = {"event_id": cfg.events + 1, "day_of_week": 7, "week_of_year": 53}
    for name, card in cal_cards.items():
        out[f"emb_{name}"] = card * emb_size(card)

    static_concat = sum(static_dims.values())
    out["proj_static"] = static_concat * d + d

    # ---- input projections -----------------------------------------------
    # One token per patch position per branch. A patch is `patch_days` days.
    pd_ = cfg.patch_days
    out["proj_demand_patch"] = (pd_ * 1) * d + d              # daily demand in the patch
    out["proj_demand_extra"] = 2 * d + d                       # weekly aggregate, lagged local level
    out["proj_commercial_patch"] = (cfg.channels * pd_) * d + d
    out["proj_channel_tokens"] = cfg.channels * (pd_ * d + d)  # one token per channel for channel attention
    out["proj_event_patch"] = 4 * d + d                        # days_to_event, in_window, region, prior-year
    out["missing_tokens"] = cfg.mask_groups * d                # learned "absent" embedding per channel group
    out["router_tokens"] = cfg.routers * d

    # ---- attention stack --------------------------------------------------
    sa = sa_layer_params(d)
    out["attn_channel_module"] = sa                            # role 3, over channel tokens
    out["attn_event_module"] = sa                              # role 4, over the event set
    out["attn_group"] = cfg.stages * sa                        # cross-branch, per patch position
    out["attn_time"] = cfg.stages * sa                         # cross-time, over patch positions
    out["attn_siblings"] = sa                                  # role 6
    out["attn_substitution"] = sa                              # role 7
    out["decoder"] = cfg.decoder_layers * xa_layer_params(d)

    # ---- output heads -----------------------------------------------------
    # p, mu, k per grain; dispersion is pooled from static attributes, not a table.
    out["heads_distribution"] = cfg.grains * 3 * (d + 1)
    out["heads_share"] = d + 1

    # ---- totals -----------------------------------------------------------
    factorised = {k: v for k, v in out.items() if k != "emb_joint_series"}
    total = sum(factorised.values())

    # ---- activations ------------------------------------------------------
    P = cfg.patches                     # patch positions on the time axis
    per_layer = {
        "channel module": act_floats(P, cfg.channels, d, h) if cfg.channel_per_patch
                          else act_floats(1, cfg.channels, d, h),
        "event module": act_floats(P, cfg.events_in_window, d, h),
        "group attention (x stages)": cfg.stages * act_floats(P, cfg.branch_tokens, d, h),
        "time attention (x stages)": cfg.stages * act_floats(1, P, d, h),
        "decoder (x layers)": cfg.decoder_layers * act_floats(1, cfg.horizon + P, d, h),
    }
    act_per_series = sum(per_layer.values())

    # cross-series roles are amortised over the group they serve, not per series
    sib = act_floats(1, cfg.branches_per_batch + cfg.routers, d, h)
    sub = act_floats(1, cfg.neighbours + 1, d, h)
    act_cross = sib + sub

    return {
        "params": out,
        "total_params": total,
        "joint_params": total
            - factorised.get("emb_branch", 0)
            - factorised.get("emb_sku", 0)
            + out["emb_joint_series"],
        "act_per_layer": per_layer,
        "act_per_series_floats": act_per_series,
        "act_cross_floats": act_cross,
        "static_concat": static_concat,
        "emb_dims": static_dims,
    }


def report(cfg) -> None:
    r = size(cfg)
    p, bpp = r["params"], cfg.bytes_per_float
    W = 68

    def line(k, v, unit=""):
        print(f"  {k:<40} {v:>22,}{unit}")

    print("=" * W)
    print("PARAMETER COUNT".center(W))
    print("=" * W)
    print(f"d_model={cfg.d_model}  heads={cfg.heads}  stages={cfg.stages}  "
          f"decoder_layers={cfg.decoder_layers}")
    print(f"branches={cfg.branches:,}  skus={cfg.skus:,}  channels={cfg.channels}  "
          f"patches={cfg.patches} x {cfg.patch_days}d")
    print()
    print("-- embeddings " + "-" * (W - 14))
    for k, v in p.items():
        if k.startswith("emb_") and k != "emb_joint_series":
            line(k, v)
    emb_total = sum(v for k, v in p.items() if k.startswith("emb_") and k != "emb_joint_series")
    line("embeddings subtotal", emb_total)
    print()
    print("-- projections " + "-" * (W - 15))
    for k, v in p.items():
        if k.startswith(("proj_", "missing", "router")):
            line(k, v)
    print()
    print("-- attention and heads " + "-" * (W - 23))
    for k, v in p.items():
        if k.startswith(("attn_", "decoder", "heads_")):
            line(k, v)
    print()
    line("TOTAL PARAMETERS", r["total_params"])
    print(f"  {'':<40} {r['total_params']/1e6:>21.2f}M")
    print()
    print("-- the choice that decides feasibility " + "-" * (W - 39))
    fact = p["emb_branch"] + p.get("emb_sku", 0) + p.get("emb_sku_residual", 0) + p.get("emb_sku_hashed", 0)
    line("factorised branch + SKU embeddings", fact)
    line("JOINT (branch, SKU) embedding", p["emb_joint_series"])
    print(f"  {'ratio':<40} {p['emb_joint_series']/max(fact,1):>21,.0f}x")
    print(f"  {'joint-variant total':<40} {r['joint_params']/1e9:>21.2f}B params"
          f"  -> {r['joint_params']*16/GB:,.0f} GB optimiser state")
    print()

    print("=" * W)
    print("TRAINING MEMORY".center(W))
    print("=" * W)
    n = r["total_params"]
    states = {
        "fp32 weights": n * 4,
        "fp32 gradients": n * 4,
        "Adam moment 1": n * 4,
        "Adam moment 2": n * 4,
    }
    for k, v in states.items():
        line(k, v // MB, " MB")
    opt = sum(states.values())
    line("optimiser footprint (16 bytes/param)", opt // MB, " MB")
    print()
    print("-- activations retained for backward, per series " + "-" * (W - 49))
    for k, v in r["act_per_layer"].items():
        line(k, int(v * bpp) // 1024, " KB")
    per_series_b = r["act_per_series_floats"] * bpp
    line("per series, all layers", int(per_series_b) // 1024, " KB")
    print()
    print("-- scaling to a batch " + "-" * (W - 22))
    for bs in cfg.batch_sizes:
        act = per_series_b * bs + r["act_cross_floats"] * bpp * (bs / cfg.branches_per_batch)
        total = opt + act
        head = f"batch {bs:>5} series"
        print(f"  {head:<40} {act/GB:>9.2f} GB act  {total/GB:>7.2f} GB total"
              f"   {'FITS 16GB T4' if total < 14*GB else 'EXCEEDS 16GB T4':>16}"
              f"   {'fits 8GB P4' if total < 6.5*GB else 'exceeds 8GB P4':>15}")
    print()
    print("  headroom assumed: ~2 GB for CUDA context, fragmentation and cuDNN workspace")

    # ---- throughput: the constraint that actually binds ------------------
    print()
    print("=" * W)
    print("THROUGHPUT AND EPOCH TIME".center(W))
    print("=" * W)
    d, P = cfg.d_model, cfg.patches   # heads do not enter the FLOP count below

    def layer_flops(n_groups, g):
        T = n_groups * g
        return 24 * T * d * d + 4 * n_groups * g * g * d

    fwd = (
        (layer_flops(P, cfg.channels) if cfg.channel_per_patch else layer_flops(1, cfg.channels))
        + layer_flops(P, cfg.events_in_window)
        + cfg.stages * layer_flops(P, cfg.branch_tokens)
        + cfg.stages * layer_flops(1, P)
        + cfg.decoder_layers * layer_flops(1, cfg.horizon + P)
    )
    step_flops = 3 * fwd                      # forward + backward
    line("forward FLOPs per sample", int(fwd))
    line("fwd+bwd FLOPs per sample", int(step_flops))
    print()

    pairs = cfg.branches * cfg.skus
    active = int(pairs * cfg.density)
    samples = active * cfg.origins
    line("possible (branch, SKU) pairs", pairs)
    line(f"active at density {cfg.density:.0%}", active)
    line(f"x {cfg.origins} forecast origins = samples/epoch", samples)
    print()
    line("incumbent series (company x SKU, weekly)", cfg.skus)
    print(f"  {'series-count multiplier at branch grain':<40} {active/cfg.skus:>21,.0f}x")
    print()
    total_flops = samples * step_flops
    print(f"  {'FLOPs per epoch':<40} {total_flops:>21.3e}")
    for name, tflops in cfg.accelerators:
        secs = total_flops / (tflops * 1e12)
        h_ = secs / 3600
        unit = f"{h_:,.1f} h" if h_ < 72 else f"{h_/24:,.1f} days"
        print(f"  {name:<40} {unit:>21}   per epoch")
    print()
    print("  effective TFLOPS assume ~15-25% of peak: many small kernels, short sequences,")
    print("  and embedding gathers rather than dense matmul dominate the step.")
    print("=" * W)


def main() -> None:
    ap = argparse.ArgumentParser()
    # cardinalities -- the dominant unknown; override from the warehouse
    ap.add_argument("--branches", type=int, default=1000)
    ap.add_argument("--skus", type=int, default=20000)
    ap.add_argument("--brands", type=int, default=2000)
    ap.add_argument("--categories", type=int, default=300)
    ap.add_argument("--pack-sizes", type=int, default=200)
    ap.add_argument("--branch-types", type=int, default=20)
    ap.add_argument("--branch-groups", type=int, default=50)
    ap.add_argument("--events", type=int, default=30)
    # architecture
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--stages", type=int, default=2, help="alternating (group, time) pairs")
    ap.add_argument("--decoder-layers", type=int, default=2)
    ap.add_argument("--channels", type=int, default=30, help="commercial + covariate channels")
    ap.add_argument("--grains", type=int, default=3)
    ap.add_argument("--routers", type=int, default=16)
    ap.add_argument("--neighbours", type=int, default=20, help="top-k substitution graph")
    ap.add_argument("--branch-tokens", type=int, default=4)
    ap.add_argument("--mask-groups", type=int, default=4)
    ap.add_argument("--events-in-window", type=int, default=5)
    ap.add_argument("--branches-per-batch", type=int, default=64)
    # time axis
    ap.add_argument("--no-patch", action="store_true",
                    help="attend over 364 raw daily steps instead of 52 weekly patches")
    ap.add_argument("--channel-per-patch", action="store_true",
                    help="run channel attention at every patch position instead of once")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--bytes-per-float", type=int, default=4)
    ap.add_argument("--density", type=float, default=0.30,
                    help="fraction of (branch, SKU) pairs that are actually ranged and selling")
    ap.add_argument("--origins", type=int, default=52,
                    help="forecast origins sampled per series per epoch")
    ap.add_argument("--sku-rep", default="free",
                    choices=["free", "attributes", "residual", "hashed", "text", "text+residual"],
                    help="how SKU identity is represented")
    ap.add_argument("--sku-residual-dim", type=int, default=16)
    ap.add_argument("--sku-buckets", type=int, default=4096)
    ap.add_argument("--text-dim", type=int, default=384, help="frozen sentence-embedding width")
    cfg = ap.parse_args()

    cfg.patch_days = 1 if cfg.no_patch else 7
    cfg.patches = 364 if cfg.no_patch else 52
    cfg.batch_sizes = [32, 128, 512, 2048]
    cfg.accelerators = [
        ("T4  @  2.5 TFLOPS eff (fp32)", 2.5),
        ("T4  @ 10   TFLOPS eff (fp16 tensor)", 10.0),
        ("L4  @ 30   TFLOPS eff (fp16 tensor)", 30.0),
        ("A100 @ 120 TFLOPS eff (fp16 tensor)", 120.0),
    ]
    report(cfg)


if __name__ == "__main__":
    main()
