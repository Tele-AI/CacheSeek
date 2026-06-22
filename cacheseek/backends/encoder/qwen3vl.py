from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

if TYPE_CHECKING:
    from PIL import Image

try:
    import torch as _torch
except ModuleNotFoundError:
    _torch = None

try:
    import numpy as _np
except ModuleNotFoundError:
    _np = None

_QWEN3_VL_EMBEDDER_MODULES = [
    "cacheseek.reuse.approximate.models_src.models.qwen3_vl_embedding",
]

_QWEN3_VL_RERANKER_MODULES = [
    "cacheseek.reuse.approximate.models_src.models.qwen3_vl_reranker",
]


def _try_import_symbol(
    module_candidates: List[str],
    symbol_name: str,
    label: str,
) -> Optional[Any]:
    import_failures: list[str] = []
    for module_path in module_candidates:
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            missing_name = exc.name or ""
            if missing_name == module_path or module_path.startswith(f"{missing_name}."):
                continue
            message = (
                f"{module_path} failed because dependency {missing_name!r} is missing: {exc}"
            )
            logger.warning(f"{label} import failed: {message}")
            import_failures.append(message)
            continue
        except Exception as exc:
            logger.exception(f"{label} import failed for {module_path}: {exc}")
            import_failures.append(
                f"{module_path} failed with {type(exc).__name__}: {exc}"
            )
            continue
        symbol = getattr(module, symbol_name, None)
        if symbol is None:
            logger.warning(f"{label} missing symbol {symbol_name} in {module_path}")
            continue
        return symbol
    if import_failures:
        raise ImportError(f"{label} import failed: {'; '.join(import_failures)}")
    return None


def _process_inputs(processor: object, inputs: object) -> object:
    processor_type = type(processor).__name__
    has_process = hasattr(processor, "process")
    is_callable = callable(processor)
    try:
        if has_process:
            return processor.process(inputs)
        if is_callable:
            return processor(inputs)
    except Exception as exc:
        logger.exception(
            "processor invocation failed processor_type={} has_process={} callable={} err={}",
            processor_type,
            has_process,
            is_callable,
            exc,
        )
        raise RuntimeError(
            "processor invocation failed "
            f"processor_type={processor_type} "
            f"has_process={has_process} callable={is_callable} "
            f"err_type={type(exc).__name__} err={exc}"
        ) from exc
    raise TypeError(f"processor is neither callable nor provides process() processor_type={processor_type}")


def _extract_first_vector(vectors: Any) -> List[float]:
    if vectors is None:
        return []
    if isinstance(vectors, list):
        if not vectors:
            return []
        first = vectors[0]
        if isinstance(first, (int, float)):
            return [float(value) for value in vectors]
        if _torch is not None and isinstance(first, _torch.Tensor):
            return first.detach().cpu().tolist()
        return list(first)
    if _torch is not None and isinstance(vectors, _torch.Tensor):
        if vectors.numel() == 0:
            return []
        if vectors.dim() == 1:
            return vectors.detach().cpu().tolist()
        return vectors[0].detach().cpu().tolist()
    if _np is not None and isinstance(vectors, _np.ndarray):
        if vectors.size == 0:
            return []
        if vectors.ndim == 1:
            return vectors.tolist()
        return vectors[0].tolist()
    return []


class Qwen3VLEncoder:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-Embedding-2B",
        instruction: str = "Represent the user's input",
        max_frames: int = 16,
        fps: float = 1.0,
        torch_dtype: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        device_id: Optional[int] = None,
        embedder: Optional[object] = None,
    ) -> None:
        self.model_path = model_path
        self.instruction = instruction
        self.max_frames = int(max_frames)
        self.fps = float(fps)
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.device_id = device_id
        self._embedder = embedder
        self._embedder_init_error: Optional[BaseException] = None
        self._embedder_init_attempted = embedder is not None
        if self._embedder is None:
            self._get_embedder()

    def encode(self, prompt: str) -> List[float]:
        return self._encode_inputs([{"text": prompt or "", "instruction": self.instruction}])

    def encode_video(
        self,
        frames: List["Image.Image"],
        prompt: Optional[str] = None,
    ) -> List[float]:
        if not frames:
            return []
        item: Dict[str, object] = {
            "video": frames,
            "instruction": self.instruction,
        }
        if prompt is not None:
            item["text"] = prompt or ""
        return self._encode_inputs([item])

    def decompose_prompt(self, prompt: str) -> Dict[str, str]:
        return {"whole": prompt or ""}

    def _encode_inputs(self, inputs: List[Dict[str, object]]) -> List[float]:
        embedder = self._get_embedder()
        if embedder is None:
            return []
        return _extract_first_vector(_process_inputs(embedder, inputs))

    def _get_embedder(self) -> Optional[object]:
        if self._embedder is not None:
            return self._embedder
        if self._embedder_init_error is not None:
            raise self._embedder_init_error
        if self._embedder_init_attempted:
            return None
        self._embedder_init_attempted = True

        try:
            embedder_cls = _try_import_symbol(
                _QWEN3_VL_EMBEDDER_MODULES,
                "Qwen3VLEmbedder",
                "Qwen3VLEncoder",
            )
        except ImportError as exc:
            self._embedder_init_error = exc
            raise
        if embedder_cls is None:
            self._embedder_init_error = ImportError(
                f"Qwen3VLEncoder embedder not found in candidates={_QWEN3_VL_EMBEDDER_MODULES}"
            )
            raise self._embedder_init_error

        init_kwargs: Dict[str, object] = {
            "model_name_or_path": self.model_path,
            "fps": self.fps,
            "max_frames": self.max_frames,
        }
        try:
            params = inspect.signature(embedder_cls.__init__).parameters
            if "torch_dtype" in params and self.torch_dtype is not None:
                init_kwargs["torch_dtype"] = self.torch_dtype
            if "attn_implementation" in params and self.attn_implementation is not None:
                init_kwargs["attn_implementation"] = self.attn_implementation
            if self.device_id is not None:
                did = int(self.device_id)
                if "device_id" in params:
                    init_kwargs["device_id"] = did
                elif "device" in params:
                    init_kwargs["device"] = "cpu" if did < 0 else f"cuda:{did}"
        except (TypeError, ValueError) as exc:
            logger.exception(
                "Qwen3VLEncoder could not inspect __init__ signature embedder_cls={} err_type={} err={}",
                getattr(embedder_cls, "__name__", repr(embedder_cls)),
                type(exc).__name__,
                exc,
            )

        try:
            self._embedder = embedder_cls(**init_kwargs)
        except Exception as exc:
            logger.exception(
                "Qwen3VLEncoder init failed model_path={} embedder_cls={} err={}",
                self.model_path,
                getattr(embedder_cls, "__name__", repr(embedder_cls)),
                exc,
            )
            self._embedder_init_error = RuntimeError(
                "Qwen3VLEncoder init failed "
                f"model_path={self.model_path} "
                f"embedder_cls={getattr(embedder_cls, '__name__', repr(embedder_cls))} "
                f"type={type(exc).__name__} err={exc}"
            )
            raise self._embedder_init_error from exc
        return self._embedder


class Qwen3VLReranker:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-Reranker-8B",
        instruction: str = "Retrieval relevant image or text with user's query",
        fps: float = 1.0,
        device_id: Optional[int] = None,
        batch_size: int = 2,
        torch_dtype: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        reranker: Optional[object] = None,
    ) -> None:
        self.model_path = model_path
        self.instruction = instruction
        self.fps = float(fps)
        self.device_id = device_id
        self.batch_size = max(1, int(batch_size or 1))
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self._reranker = reranker
        self._reranker_init_error: Optional[BaseException] = None
        self._reranker_init_attempted = reranker is not None
        if self._reranker is None:
            self._init_reranker()

    def score_mm(self, query: Dict[str, object], documents: List[Dict[str, object]]) -> List[float]:
        if not isinstance(query, dict) or not documents:
            return []
        self._init_reranker()
        if self._reranker is None:
            return []
        scores = self._score_with_reranker_mm(self._reranker, query, documents)
        return self._normalize_scores(scores, len(documents))

    def _init_reranker(self) -> None:
        if self._reranker is not None:
            return
        if self._reranker_init_error is not None:
            raise self._reranker_init_error
        if self._reranker_init_attempted:
            return
        self._reranker_init_attempted = True

        if not Path(self.model_path).exists():
            logger.warning(
                "Qwen3VLReranker model_path is not local: {!r}. "
                "If you intend to load offline, set HF_HUB_OFFLINE=1 and "
                "TRANSFORMERS_OFFLINE=1 in the environment before launching.",
                self.model_path,
            )

        try:
            reranker_cls = _try_import_symbol(
                _QWEN3_VL_RERANKER_MODULES,
                "Qwen3VLReranker",
                "Qwen3VLReranker",
            )
        except ImportError as exc:
            self._reranker_init_error = exc
            raise
        if reranker_cls is None:
            self._reranker_init_error = ImportError(
                f"Qwen3VLReranker implementation not found in candidates={_QWEN3_VL_RERANKER_MODULES}"
            )
            raise self._reranker_init_error

        init_kwargs: Dict[str, object] = {"model_name_or_path": self.model_path}
        try:
            params = inspect.signature(reranker_cls.__init__).parameters
            if "batch_size" in params:
                init_kwargs["batch_size"] = self.batch_size
            if "torch_dtype" in params and self.torch_dtype is not None:
                init_kwargs["torch_dtype"] = self.torch_dtype
            if "attn_implementation" in params and self.attn_implementation is not None:
                init_kwargs["attn_implementation"] = self.attn_implementation
            if self.device_id is not None:
                did = int(self.device_id)
                if "device_id" in params:
                    init_kwargs["device_id"] = did
                elif "device" in params:
                    init_kwargs["device"] = "cpu" if did < 0 else f"cuda:{did}"
        except (TypeError, ValueError) as exc:
            logger.exception(
                "Qwen3VLReranker could not inspect __init__ signature reranker_cls={} err_type={} err={}",
                getattr(reranker_cls, "__name__", repr(reranker_cls)),
                type(exc).__name__,
                exc,
            )

        try:
            self._reranker = reranker_cls(**init_kwargs)
        except Exception as exc:
            logger.exception(
                "Qwen3VLReranker init failed model_path={} reranker_cls={} err={}",
                self.model_path,
                getattr(reranker_cls, "__name__", repr(reranker_cls)),
                exc,
            )
            self._reranker_init_error = RuntimeError(
                "Qwen3VLReranker init failed "
                f"model_path={self.model_path} "
                f"reranker_cls={getattr(reranker_cls, '__name__', repr(reranker_cls))} "
                f"type={type(exc).__name__} err={exc}"
            )
            raise self._reranker_init_error from exc

    def _score_with_reranker_mm(
        self,
        reranker: object,
        query: Dict[str, object],
        documents: List[Dict[str, object]],
    ) -> Optional[object]:
        if not hasattr(reranker, "process"):
            raise AttributeError(f"Qwen3VLReranker backend is missing process() type={type(reranker).__name__}")
        inputs = {
            "instruction": self.instruction,
            "query": query,
            "documents": documents,
            "fps": self.fps,
        }
        try:
            return reranker.process(inputs)
        except TypeError as exc:
            logger.exception(
                "Qwen3VLReranker.process rejected multimodal payload reranker_type={} err={}",
                type(reranker).__name__,
                exc,
            )
            raise RuntimeError(
                "Qwen3VLReranker.process rejected multimodal payload "
                f"reranker_type={type(reranker).__name__} "
                f"err_type={type(exc).__name__} err={exc}"
            ) from exc

    def _normalize_scores(self, scores: Optional[object], expected_len: int) -> List[float]:
        if scores is None:
            return []
        if _torch is not None and isinstance(scores, _torch.Tensor):
            scores = scores.detach().cpu().tolist()
        if _np is not None and isinstance(scores, _np.ndarray):
            scores = scores.tolist()
        if not isinstance(scores, list) or not scores:
            return []
        if isinstance(scores[0], dict):
            values = []
            for item in scores:
                if not isinstance(item, dict):
                    continue
                if "score" in item:
                    values.append(float(item["score"]))
                elif "relevance" in item:
                    values.append(float(item["relevance"]))
                elif "logit" in item:
                    values.append(float(item["logit"]))
            return values
        if isinstance(scores[0], (list, tuple)) and len(scores[0]) == 2 and isinstance(scores[0][0], int):
            ordered = [float("-inf")] * expected_len
            for idx, value in scores:
                if 0 <= int(idx) < expected_len:
                    ordered[int(idx)] = float(value)
            return ordered
        return [float(item) for item in scores]
