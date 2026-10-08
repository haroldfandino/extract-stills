"""Console entry point for a frozen distribution."""

from multiprocessing import freeze_support


if __name__ == "__main__":
    freeze_support()
    from stills_tool.cli import main

    raise SystemExit(main())
