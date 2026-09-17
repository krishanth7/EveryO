"""Allow ``python -m everyo`` to run the CLI."""

from __future__ import annotations

from everyo.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
