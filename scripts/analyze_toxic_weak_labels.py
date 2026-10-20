#!/usr/bin/env python
"""Backward-compatible entry point for the generic weak-label audit."""
if __package__:
    from .analyze_weak_labels import main
else:
    from analyze_weak_labels import main


if __name__ == "__main__":
    main()
