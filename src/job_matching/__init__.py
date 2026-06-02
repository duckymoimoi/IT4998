"""Job matching application package."""

import sys


def _configure_utf8_stdio():
    """Keep Vietnamese CLI text printable on Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


_configure_utf8_stdio()
