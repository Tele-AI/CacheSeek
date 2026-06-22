# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Import a symbol from a Python file by path (e.g. CACHE_CONFIG / PPL_CONFIG
defined in a ppl file).

Kept here so cacheseek's own assembly chain does not depend on whether
telefuser is installed; telefuser is imported only on the code paths that
actually wire into the engine.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys


def import_function_from_file(file_path: str, function_name: str):
    """Import a symbol from a Python file by path.

    The module is registered in ``sys.modules`` under a path-derived unique
    name so files with the same basename do not overwrite each other; on
    load failure the entry is removed to avoid leaking partial state.
    """
    abs_path = os.path.abspath(file_path)
    basename = os.path.splitext(os.path.basename(abs_path))[0]
    path_hash = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:12]
    module_name = f"_cacheseek_ppl_{basename}_{path_hash}"

    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build import spec for file_path={abs_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return getattr(module, function_name)
