#!/usr/bin/env python3
"""Copper Ore collector: thin wrapper around the generic DarkerDB v2 collector (src/api/collect.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.api.collect import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["--item", "Copper Ore"] + sys.argv[1:]))
