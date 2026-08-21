"""Entry point for the corrected 2D application."""

from __future__ import annotations

import sys


def main() -> int:
    """Launch the Tkinter suite.

    Kept thin so the scientific core can be imported and tested without ever
    creating a GUI.
    """
    from ..ui.main_window import run

    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
