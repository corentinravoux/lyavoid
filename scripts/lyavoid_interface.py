#!/usr/bin/env python3
"""CLI entry point: run the lyavoid pipeline from an interface ``.ini`` file.

Usage: ``lyavoid_interface.py <config.ini>`` (see :func:`lyavoid.interface.main`).
"""
from lyavoid import interface
import sys

if __name__ == "__main__":
    interface.main(sys.argv[1])
