"""A four-line test runner: ``python -m tests``.

The project depends on nothing but numpy and Pillow, and adding a test
framework to check that would be silly.  Every module named ``test_*.py`` in
this package is imported, every callable named ``test_*`` is called, and the
first failure stops the run with its traceback.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import time
import traceback


def _cases():
    import tests as pkg

    for info in pkgutil.iter_modules(pkg.__path__):
        if not info.name.startswith("test_"):
            continue
        module = importlib.import_module(f"tests.{info.name}")
        for name in sorted(vars(module)):
            fn = getattr(module, name)
            if name.startswith("test_") and callable(fn):
                yield f"{info.name}.{name}", fn


def main(argv=None):
    wanted = list(argv if argv is not None else sys.argv[1:])
    failures = 0
    total = 0
    for label, fn in _cases():
        if wanted and not any(w in label for w in wanted):
            continue
        total += 1
        started = time.time()
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL  {label}\n")
            traceback.print_exc()
            break
        print(f"ok    {label}  ({time.time() - started:.1f}s)")
    passed = total - failures
    print(f"\n{passed}/{total} passed")
    return 1 if failures else 0
