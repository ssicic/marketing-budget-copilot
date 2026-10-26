"""Marketing Budget Copilot."""

import os
import sys
from pathlib import Path

# PyTensor (PyMC's backend) passes "-ld64" to clang on macOS >= 15, which recent
# Xcode rejects, so every compiled model fails to build. Point PyTensor at a wrapper
# that drops the flag. This must run before pytensor is imported, so import mbc first.
_CXX_WRAPPER = Path(__file__).resolve().parents[2] / "scripts" / "macos-clang++"
if sys.platform == "darwin" and _CXX_WRAPPER.exists():
    _flags = os.environ.get("PYTENSOR_FLAGS", "")
    if "cxx=" not in _flags:
        os.environ["PYTENSOR_FLAGS"] = ",".join(filter(None, [_flags, f"cxx={_CXX_WRAPPER}"]))
