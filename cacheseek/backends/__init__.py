# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""cacheseek.backends — pluggable backend implementations(design D8 §8 hierarchy).

Provides a stable namespace grouping KV / Vector / Metadata / Audit backends.
Interfaces live under cacheseek.service.interfaces / cacheseek.stores.base (single source of truth);
this package re-exports them at the design-aligned import paths.
"""
