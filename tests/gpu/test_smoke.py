"""GPU smoke tests: the installed torch can use the pod's GPU.

scripts/setup_pod.sh runs them with `pytest -m gpu -k smoke`. Elsewhere they
are skipped, like every gpu test.
"""

import importlib

import pytest
import torch


@pytest.mark.gpu
def test_cuda_is_available() -> None:
    assert torch.cuda.is_available(), (
        f"torch {torch.__version__} (CUDA {torch.version.cuda}) sees no GPU"
    )


@pytest.mark.gpu
def test_a_small_matrix_multiply_on_the_gpu_gives_the_right_answer() -> None:
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")
    product = a @ b
    assert product.device.type == "cuda"
    assert product.cpu().tolist() == [[19.0, 22.0], [43.0, 50.0]]


# The fast kernels transformers looks for to run Qwen3.5's linear-attention
# layers. If one fails to import, it quietly uses much slower PyTorch code.
KERNELS = {
    "causal_conv1d": ["causal_conv1d_fn", "causal_conv1d_update"],
    "fla.ops.gated_delta_rule": [
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
    ],
}


@pytest.mark.gpu
@pytest.mark.parametrize("module", sorted(KERNELS))
def test_the_qwen3_5_kernels_import(module: str) -> None:
    imported = importlib.import_module(module)
    for name in KERNELS[module]:
        assert callable(getattr(imported, name, None)), f"{module} lacks {name}"
