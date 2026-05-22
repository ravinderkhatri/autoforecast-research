"""Compatibility entry point for the time-series installed-package acceptance gate."""

from scripts.verify_timeseries_release import main

if __name__ == "__main__":
    raise SystemExit(main())
