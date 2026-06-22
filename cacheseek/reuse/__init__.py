"""reuse — the reuse-semantics layer: two coexisting hit semantics over the
same cache concept.

  exact_prefix/   Exact prefix (hash trie): a hit means the exact same path;
                  lossless fast-forward.
  approximate/    Approximate recall (embedding ANN): a hit means "close
                  enough"; lossy skip_step resumption.

Both share the same skeleton: lookup(request) -> hit -> materialize resume
state -> generate -> ingest write-back. They differ only in the
KEY / index / VALUE / resume protocol.
"""
