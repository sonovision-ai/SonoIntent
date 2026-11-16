#!/usr/bin/env python3
"""Build the official SonoIntent train/val/test split from rendered V1 records."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sonointent.build.build_split import main

if __name__ == "__main__":
    raise SystemExit(main())
