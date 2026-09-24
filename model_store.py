"""Keeps at most MAX_RESIDENT_MODELS heavy models in RAM at once (default 1)."""
from __future__ import annotations

import ctypes
import gc
import os
import threading
from collections import OrderedDict
from typing import Callable

MAX_RESIDENT = int(os.getenv("MAX_RESIDENT_MODELS", "1"))
_lock = threading.RLock()
_models: "OrderedDict[str, object]" = OrderedDict()


def _release_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # Linux: hand freed memory back to the OS
    except Exception:  # noqa: BLE001
        pass


def get_model(key: str, loader: Callable[[], object]):
    with _lock:
        if key in _models:
            _models.move_to_end(key)
            return _models[key]
        evicted = False
        while _models and len(_models) >= MAX_RESIDENT:
            _models.popitem(last=False)
            evicted = True
        if evicted:
            _release_memory()  # free the old model BEFORE loading the new one
        obj = loader()
        _models[key] = obj
        return obj