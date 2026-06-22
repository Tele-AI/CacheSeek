# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT / "examples" / "approximate_reuse" / "service" / "start_wan22_service.sh"
)
pytestmark = pytest.mark.smoke


class StartWan22ServiceScriptTest(unittest.TestCase):
    def test_dry_run_prints_script_owned_cache_config(self) -> None:
        result = subprocess.run(
            ["bash", str(SCRIPT), "--preset", "s1_fresh", "--dry-run"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
            env={**__import__("os").environ, "PY_SVC": sys.executable},
        )

        out = result.stdout
        self.assertIn("server_config.enable_latent_cache=True", out)
        self.assertIn("CACHE_CONFIG", out)
        self.assertIn("enable_latent_cache=True", out)
        self.assertIn("cache_mode=read_write", out)
        self.assertIn("latent_cache_dir=", out)
        self.assertIn("kv_store_type=local_file", out)
        self.assertIn("fluxon_config_path=", out)
        # s1_fresh is the daemon-free smoke preset: faiss (no qdrant server needed).
        self.assertIn("vector_store_type=faiss", out)
        self.assertIn("video_embedding_model_path=", out)
        self.assertIn("rerank_model_path=", out)
        self.assertIn("run_server(", out)
        self.assertNotIn("TELEFUSER_KV_STORE_TYPE=", out)
        self.assertNotIn("TELEFUSER_VECTOR_STORE_TYPE=", out)
        self.assertNotIn("TELEFUSER_LATENT_CACHE_DIR=", out)
        self.assertNotIn("FLUXON_CONFIG_PATH=", out)


if __name__ == "__main__":
    unittest.main()
