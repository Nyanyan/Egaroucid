#!/usr/bin/env python3
"""Production entry point for the 500-set strength tournament.

All arguments and defaults live in ``battle_blend_strength.py``. Keeping this
file as a thin alias avoids the former pair of parsers drifting apart.
"""

from battle_blend_strength import main, make_argparser


if __name__ == "__main__":
    main()
