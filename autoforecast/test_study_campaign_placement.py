"""Literal Ray placement runs without a live module-scoped Spark session."""

import json
import sys

from autoforecast import study_campaign
from autoforecast.test_study_campaign import _RayRecorder


def test_campaign_label_placement_is_literal(tmp_path, monkeypatch):
    import ray
    from ray.cluster_utils import Cluster
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    VALUES = [
        "!forecast",
        "in(forecast)",
        "in(forecast,other)",
        "in(forecast, other)",
        "!in(forecast)",
        "!in(forecast,other)",
        "!in(forecast, other)",
        "(forecast)",
        "(forecast,other)",
        "!(forecast)",
        "!(forecast,other)",
        "notin(forecast)",
        "!!forecast",
        "in()",
        "in(forecast,)",
        "in(forecast,  other)",
        "IN(forecast)",
        "in (forecast)",
        "forecast!",
        "forecast",
        "in",
        "notin",
    ]

    CASES = [(f"k{index}", value, value) for index, value in enumerate(VALUES)] + [
        (key, "forecast", "key:" + key)
        for key in [
            "pool name",
            "pool:region",
            "a/b/c",
            "/pool",
            "pool/",
            "a" * 64,
            "a_b/name",
            "example.com/pool",
        ]
    ]

    cluster = Cluster()
    tasks = {}
    outcomes = {}
    try:
        labels = {key: value for key, value, _ in CASES}
        cluster.add_node(
            num_cpus=1,
            memory=512 * 1024**2,
            object_store_memory=80 * 1024**2,
            include_dashboard=False,
            labels=labels,
        )
        cluster.add_node(
            num_cpus=1,
            memory=512 * 1024**2,
            object_store_memory=80 * 1024**2,
            labels=dict.fromkeys(labels, "unrelated"),
        )
        ray.init(address=cluster.address, log_to_driver=False)
        nodes = [node for node in ray.nodes() if node["Alive"]]
        exact = next(node["NodeID"] for node in nodes if node["Labels"]["k0"] == "!forecast")
        other = next(node["NodeID"] for node in nodes if node["Labels"]["k0"] == "unrelated")

        @ray.remote
        class Hold:
            def ping(self):
                return True

        holder = Hold.options(
            num_cpus=1, scheduling_strategy=NodeAffinitySchedulingStrategy(exact, soft=False)
        ).remote()
        ray.get(holder.ping.remote(), timeout=30)

        @ray.remote
        def where():
            import ray

            return ray.get_runtime_context().get_node_id()

        # Warm the second worker to avoid confusing process startup with scheduling.
        assert (
            ray.get(
                where.options(
                    num_cpus=1,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(other, soft=False),
                ).remote(),
                timeout=30,
            )
            == other
        )
        for index, (label_key, value, case) in enumerate(CASES):
            folder = tmp_path / str(index)
            folder.mkdir()
            request = {
                "models": [
                    {
                        "id": "m",
                        "cpus": 1,
                        "gpus": 0,
                        "memory_gib": 0.25,
                        "node_labels": {label_key: value},
                    }
                ],
                "execution": "cluster",
                "ray_address": cluster.address,
                "max_parallel_models": 1,
                "timeout_seconds": 30,
                "final_evaluation": "all_models",
            }
            fake = _RayRecorder([])
            fake.nodes = lambda: nodes
            with monkeypatch.context() as patch:
                patch.setitem(sys.modules, "ray", fake)
                patch.setattr(study_campaign, "_read", lambda *args, request=request: request)
                patch.setattr(
                    study_campaign, "_load_model_result", lambda *args, **kwargs: {"index": 0}
                )
                study_campaign._run_phase(
                    "unused", "unused", "validation", str(folder / "campaign"), str(folder)
                )
            assert len(fake.submissions) == 1, "literal live-node admission unexpectedly failed"
            options = fake.submissions[0][1]
            try:
                tasks[case] = where.options(**options).remote()
            except ValueError as error:
                outcomes[case] = "REJECTED_AFTER_ADMISSION: " + str(error).splitlines()[0]
        refs = list(tasks.values())
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=3)
        for value, task in tasks.items():
            if task in ready:
                actual = ray.get(task)
                outcomes[value] = (
                    "WRONG_NODE" if actual != exact else "UNEXPECTED_EXACT_NODE_WHILE_RESERVED"
                )
        ray.kill(holder)
        pending = [task for task in refs if task not in ready]
        if pending:
            later, _ = ray.wait(pending, num_returns=len(pending), timeout=30)
            for value, task in tasks.items():
                if value in outcomes:
                    continue
                if task in later:
                    actual = ray.get(task)
                    outcomes[value] = "EXACT_NODE" if actual == exact else "WRONG_NODE"
                else:
                    outcomes[value] = "UNSCHEDULABLE_WITH_EXACT_NODE_FREE"
        bad = {value: outcome for value, outcome in outcomes.items() if outcome != "EXACT_NODE"}
        assert not bad, "Campaign violated exact literal placement: " + json.dumps(
            bad, sort_keys=True
        )
    finally:
        for task in tasks.values():
            if ray.is_initialized():
                ray.cancel(task, force=True)
        ray.shutdown()
        cluster.shutdown()
