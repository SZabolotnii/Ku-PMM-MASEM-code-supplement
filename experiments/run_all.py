"""One-command local reproduction driver for the current PMM-MASEM evidence."""

from __future__ import annotations

from experiments import generate_tables_figures, run_known_dgp_mc, run_resampling_proxy


def main() -> None:
    run_known_dgp_mc.main()
    run_resampling_proxy.main()
    generate_tables_figures.main()


if __name__ == "__main__":
    main()

