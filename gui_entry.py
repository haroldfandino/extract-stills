"""Desktop entry point for a frozen distribution."""

from multiprocessing import freeze_support
import sys


if __name__ == "__main__":
    freeze_support()
    if sys.argv[1:2] == ["--legacy-worker"]:
        from stills_tool.cli import main
    else:
        from stills_tool.gui import main

    raise SystemExit(main())
