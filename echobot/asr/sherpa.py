from __future__ import annotations

from typing import Any


def sherpa_dependency_error_message() -> str | None:
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return "sherpa-onnx 不可用，请先安装: pip install sherpa-onnx"
    return None


def load_sherpa_module() -> Any:
    try:
        import sherpa_onnx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "sherpa-onnx 不可用，请先安装: pip install sherpa-onnx"
        ) from exc
    return sherpa_onnx
