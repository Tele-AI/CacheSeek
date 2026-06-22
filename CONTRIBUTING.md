# Contributing to CacheSeek

Thanks for contributing. This guide covers the **code-style conventions** —
specifically comments, docstrings, and headers. They are maintained by
convention and review (not a docstring linter), so please follow them by example.

Run `ruff check --fix && ruff format` before pushing. Run the fast checks with
`pytest -m smoke`.

## File headers

Every `.py` file starts with the two-line SPDX header:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
```

**Adapted / ported code** adds an attribution block right under the header,
crediting the upstream project with a URL, naming its license, and pinning the
exact upstream commit when known:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project

# Adapted from <project> (<url>, <license>). Modified for CacheSeek.
```

## Docstrings

- **Google style** — `Args:` / `Returns:` / `Raises:` sections. Do **not** use
  reST/Sphinx fields (`:param:`, `:return:`, `:rtype:`).
- **Types live in annotations, not docstrings.** Don't restate a parameter's type
  in its description.
- Custom sections are welcome where they add clarity (e.g. `Conformance:`,
  `Shapes:`, `Internal updates:`).
- The whole codebase is **English-only** (comments, docstrings, and user-facing
  strings).

Document by tier — not everything needs a docstring:

| Symbol | Docstring? |
|---|---|
| Protocols / interfaces (`service/interfaces/`) | **Yes** — the full contract lives here |
| Protocol *implementations* (e.g. `FAISSVectorStore.search`) | Only if behavior deviates from the contract; otherwise skip |
| Public functions/classes with non-obvious logic | **Yes** — Google style |
| Self-evident helpers (name + signature say it all) | No |
| Vendored model code (`reuse/approximate/models_src/`) | No — mirror upstream verbatim |

Example:

```python
async def lookup(self, query: CacheQuery, ctx: Any = None) -> LookupResult:
    """Decide whether `query` hits a stored cache.

    Args:
        query: Strategy-agnostic lookup request built by the FrameworkAdapter.
        ctx: Optional strategy-specific context (e.g. the engine handle).

    Returns:
        A LookupResult; `hit` is False on a miss, with `resume_hint` / `payload`
        unset.
    """
```

## Tensor shapes

Caching KV/latent tensors is the core of this project, so **document shapes**.

- On payload / cache classes, add a `Shapes:` block listing each tensor and its
  dimensions with short named codes, and note which dimensions are **stable
  across requests** vs **per-request** — that distinction is the cross-request
  reuse invariant.
- On non-obvious tensor locals, add an inline `# (T, C, H, W)` annotation.

```python
class VideoApproxPayload:
    """Cached early-denoise latents for one donor request.

    Shapes:
        latent[step]: (B, C, T, H, W)
            B  batch (always 1 for a cached donor)        — stable
            C  latent channels                            — stable
            T  latent frames                              — stable per model profile
            H, W  latent spatial dims                     — vary per resolution
    """
```

(Lightweight prose by design — we don't use runtime-validated shape schemas.)

## Config fields

Config dataclasses (`CacheConfig` and friends) document each field with a
**PEP-257 attribute docstring** — a triple-quoted string immediately after the
field — so the documentation travels with the default:

```python
max_skip_step: int = 5
"""Upper bound on how many denoise steps a cache hit may skip.

At lookup, the largest checkpointed step `<= max_skip_step` is chosen.
"""
```

Prefer this over a trailing `#` comment for any field worth more than a few words.

## Inline comments

- Explain **why, not what.** Assume the reader knows the domain; don't narrate
  what the code plainly does.
- Densest on the subtle / hot paths (RNG alignment, ring reassembly, cache
  invariants) — that's where a future reader needs the rationale.
- Tag vocabulary:
  - `# NOTE:` — an invariant, external constraint, or non-obvious "why".
  - `# TODO(owner or #issue):` — pending work, with a clear resolution condition.
  - `# FIXME:` — known-wrong-but-works debt.
  - Avoid `# HACK` / `# XXX`.

## Tooling

- **Lint:** `ruff check` (config in `pyproject.toml`: `E, F, I, UP, B, SIM`).
- **Format:** `ruff format` (owns line length, 100 cols; formats code inside
  docstrings).
- We intentionally do **not** enforce docstring presence/format (`ruff` `D`
  codes / pydocstyle) — the conventions above are upheld by review, matching
  vLLM and SGLang.
- Vendored `models_src/` is excluded from lint and formatting.
