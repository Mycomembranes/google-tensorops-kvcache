"""Metal backend: transpiles .kernel (WGSL) to MSL, wraps with mx.fast.metal_kernel.

This backend is used on macOS with Apple Silicon (M1/M2/M3/M4).
Kernels are JIT-compiled via MLX's custom Metal kernel API.
"""

import os
import math
import struct
from typing import List, Tuple, Dict, Optional, Callable, Any

import mlx.core as mx

from ..transpiler import wgsl_to_msl, load_kernel


# Cache transpiled kernels to avoid re-transpiling
_kernel_cache: Dict[str, Any] = {}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def compile_kernel(kernel_path: str, kernel_name: str) -> dict:
    """Load a .kernel file, transpile to MSL, and compile as Metal kernel.

    Args:
        kernel_path: Path to .kernel file (WGSL syntax)
        kernel_name: Name for the compiled kernel

    Returns:
        Dict with 'kernel' (the mx.fast.metal_kernel callable),
        'input_names', 'output_names', 'workgroup_size',
        'needs_array_length'
    """
    cache_key = f"{kernel_path}:{kernel_name}"
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    source = load_kernel(kernel_path)
    msl = wgsl_to_msl(source, kernel_name)

    # For buffers that use arrayLength(&buf), the transpiler converts it to
    # buf_length. We need to add these as extra input parameters so that
    # mx.fast.metal_kernel generates `const uint& buf_length` in the
    # function signature. We pass them as uint scalar inputs at call time.
    input_names = list(msl["input_names"])
    for buf_name in msl["needs_array_length"]:
        length_name = f"{buf_name}_length"
        if length_name not in input_names:
            input_names.append(length_name)

    kernel = mx.fast.metal_kernel(
        name=kernel_name,
        input_names=input_names,
        output_names=msl["output_names"],
        source=msl["source"],
        header=msl["header"],
        ensure_row_contiguous=True,
        atomic_outputs=False,
    )

    result = {
        "kernel": kernel,
        "input_names": input_names,
        "output_names": msl["output_names"],
        "workgroup_size": msl["workgroup_size"],
        "needs_array_length": msl["needs_array_length"],
    }

    _kernel_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Kernel directory
# ---------------------------------------------------------------------------

_KERNEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kernels")


def _kernel_path(name: str) -> str:
    return os.path.join(_KERNEL_DIR, name)


# ---------------------------------------------------------------------------
# Fused RMS Norm
# ---------------------------------------------------------------------------

def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-5) -> mx.array:
    """Fused RMS normalization via Metal kernel.

    Args:
        x: Input tensor of shape (..., D)
        weight: Learnable scale of shape (D,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor of same shape as x
    """
    original_shape = x.shape
    D = x.shape[-1]
    # Flatten to 2D: (rows, D)
    x_2d = x.reshape(-1, D)
    rows = x_2d.shape[0]

    compiled = compile_kernel(
        _kernel_path("fused_rms_norm.kernel"),
        "fused_rms_norm"
    )
    kernel = compiled["kernel"]
    wgs = compiled["workgroup_size"]

    # params: vec4<f32> with params.x = width, params.y = eps
    params = mx.array([float(D), eps, 0.0, 0.0], dtype=mx.float32)

    # grid = total threads. Kernel uses one workgroup per row, with
    # wgs[0] threads per workgroup, so total threads = rows * wgs[0].
    outputs = kernel(
        inputs=[x_2d, weight, params],
        output_shapes=[x_2d.shape],
        output_dtypes=[x.dtype],
        grid=(rows * wgs[0], 1, 1),
        threadgroup=wgs,
        template=[("T", mx.float32)],
    )

    return outputs[0].reshape(original_shape)


# ---------------------------------------------------------------------------
# Fused SwiGLU
# ---------------------------------------------------------------------------

def fused_swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """Fused SwiGLU activation: SiLU(gate) * up.

    Args:
        gate: Gate activations (output of gate projection)
        up: Up activations (output of up projection)

    Returns:
        SiLU(gate) * up, same shape
    """
    assert gate.shape == up.shape, f"Shape mismatch: gate {gate.shape} vs up {up.shape}"

    n = gate.size
    gate_flat = gate.reshape(-1)
    up_flat = up.reshape(-1)

    compiled = compile_kernel(
        _kernel_path("fused_swiglu.kernel"),
        "fused_swiglu"
    )
    kernel = compiled["kernel"]
    wgs = compiled["workgroup_size"]

    # Build inputs list; append length params for arrayLength references
    inputs = [gate_flat, up_flat]
    for buf_name in compiled["needs_array_length"]:
        inputs.append(mx.array([n], dtype=mx.uint32))

    outputs = kernel(
        inputs=inputs,
        output_shapes=[(n,)],
        output_dtypes=[gate.dtype],
        grid=(_ceil_div(n, wgs[0]) * wgs[0], 1, 1),
        threadgroup=wgs,
        template=[("T", mx.float32)],
    )

    return outputs[0].reshape(gate.shape)


# ---------------------------------------------------------------------------
# Fused Rotary Embeddings
# ---------------------------------------------------------------------------

def rotary_emb(x: mx.array, cos_vals: mx.array,
               sin_vals: mx.array) -> mx.array:
    """Fused rotary position embeddings via Metal kernel.

    Args:
        x: Input tensor of shape (N, D) where N = B*T*H
        cos_vals: Cosine values of shape (N, D//2)
        sin_vals: Sine values of shape (N, D//2)

    Returns:
        Rotated tensor of shape (N, D)
    """
    assert x.ndim == 2, f"Expected 2D input, got {x.ndim}D"
    N, D = x.shape
    half_d = D // 2
    assert cos_vals.shape == (N, half_d), f"cos shape mismatch: {cos_vals.shape}"
    assert sin_vals.shape == (N, half_d), f"sin shape mismatch: {sin_vals.shape}"

    total = N * D

    compiled = compile_kernel(
        _kernel_path("fused_rotary_emb.kernel"),
        "fused_rotary_emb"
    )
    kernel = compiled["kernel"]
    wgs = compiled["workgroup_size"]

    # params: vec4<u32> with N, D, half_d
    params = mx.array([N, D, half_d, 0], dtype=mx.uint32)

    x_flat = x.reshape(-1)
    cos_flat = cos_vals.reshape(-1)
    sin_flat = sin_vals.reshape(-1)

    outputs = kernel(
        inputs=[x_flat, cos_flat, sin_flat, params],
        output_shapes=[(total,)],
        output_dtypes=[x.dtype],
        grid=(_ceil_div(total, wgs[0]) * wgs[0], 1, 1),
        threadgroup=wgs,
        template=[("T", mx.float32)],
    )

    return outputs[0].reshape(N, D)


# ---------------------------------------------------------------------------
# Fused Cross-Entropy Loss
# ---------------------------------------------------------------------------

def cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    """Fused cross-entropy loss via Metal kernel.

    Computes per-sample cross-entropy: -log_softmax(logits)[target]

    Args:
        logits: Raw logits of shape (B*T, V)
        targets: Integer target indices of shape (B*T,)

    Returns:
        Per-sample losses of shape (B*T,)
    """
    assert logits.ndim == 2, f"Expected 2D logits, got {logits.ndim}D"
    BT, V = logits.shape
    assert targets.shape == (BT,), f"Target shape mismatch: {targets.shape}"

    # Targets need to be float for the kernel (stored as f32, read as u32 inside)
    targets_f = targets.astype(mx.float32)

    compiled = compile_kernel(
        _kernel_path("fused_cross_entropy.kernel"),
        "fused_cross_entropy"
    )
    kernel = compiled["kernel"]
    wgs = compiled["workgroup_size"]

    # params: vec4<f32> with V (vocab size)
    params = mx.array([float(V), 0.0, 0.0, 0.0], dtype=mx.float32)

    # grid = total threads. Kernel uses one workgroup per row, with
    # wgs[0] threads per workgroup, so total threads = BT * wgs[0].
    outputs = kernel(
        inputs=[logits, targets_f, params],
        output_shapes=[(BT,)],
        output_dtypes=[logits.dtype],
        grid=(BT * wgs[0], 1, 1),
        threadgroup=wgs,
        template=[("T", mx.float32)],
    )

    return outputs[0]


# ---------------------------------------------------------------------------
# RMS Norm Backward
# ---------------------------------------------------------------------------

def rms_norm_backward(grad_out: mx.array, x: mx.array,
                      eps: float = 1e-5) -> mx.array:
    """Backward pass for RMS normalization via Metal kernel.

    Computes grad_x given grad_out and the original input x.
    grad_x[i] = (grad_out[i] - x_hat[i] * dot(grad_out, x_hat) / width) * inv_rms
    where x_hat = x * inv_rms, rms = sqrt(mean(x^2) + eps)

    Args:
        grad_out: Upstream gradient, same shape as x (..., D)
        x: Original input tensor of shape (..., D)
        eps: Epsilon used in the forward pass

    Returns:
        grad_x: Gradient w.r.t. x, same shape
    """
    original_shape = x.shape
    D = x.shape[-1]
    x_2d = x.reshape(-1, D)
    grad_out_2d = grad_out.reshape(-1, D)
    rows = x_2d.shape[0]

    compiled = compile_kernel(
        _kernel_path("rms_norm_backward.kernel"),
        "rms_norm_backward"
    )
    kernel = compiled["kernel"]
    wgs = compiled["workgroup_size"]

    # params: vec4<u32> with width and eps (bitcast to u32)
    eps_bits = struct.unpack("I", struct.pack("f", eps))[0]
    params = mx.array([D, eps_bits, 0, 0], dtype=mx.uint32)

    outputs = kernel(
        inputs=[grad_out_2d, x_2d, params],
        output_shapes=[x_2d.shape],
        output_dtypes=[x.dtype],
        grid=(rows * wgs[0], 1, 1),
        threadgroup=wgs,
        template=[("T", mx.float32)],
    )

    return outputs[0].reshape(original_shape)


# ---------------------------------------------------------------------------
# SiLU Backward
# ---------------------------------------------------------------------------

def silu_backward(grad_out: mx.array, x: mx.array) -> mx.array:
    """Backward pass for SiLU activation via Metal kernel.

    grad_in = grad_out * sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    Args:
        grad_out: Upstream gradient, same shape as x
        x: Original input to SiLU

    Returns:
        grad_in: Gradient w.r.t. x, same shape
    """
    original_shape = x.shape
    n = x.size
    grad_out_flat = grad_out.reshape(-1)
    x_flat = x.reshape(-1)

    compiled = compile_kernel(
        _kernel_path("silu_backward.kernel"),
        "silu_backward"
    )
    kernel = compiled["kernel"]
    wgs = compiled["workgroup_size"]

    # Build inputs; append length params for arrayLength references
    inputs = [grad_out_flat, x_flat]
    for buf_name in compiled["needs_array_length"]:
        inputs.append(mx.array([n], dtype=mx.uint32))

    outputs = kernel(
        inputs=inputs,
        output_shapes=[(n,)],
        output_dtypes=[x.dtype],
        grid=(_ceil_div(n, wgs[0]) * wgs[0], 1, 1),
        threadgroup=wgs,
        template=[("T", mx.float32)],
    )

    return outputs[0].reshape(original_shape)


# ---------------------------------------------------------------------------
# Rotary Embedding Backward
# ---------------------------------------------------------------------------

def rotary_emb_backward(grad_out: mx.array, cos_vals: mx.array,
                        sin_vals: mx.array) -> mx.array:
    """Backward pass for rotary position embeddings via Metal kernel.

    Forward was: y1 = x1*cos - x2*sin, y2 = x1*sin + x2*cos
    Backward (transpose of rotation):
        grad_x1 = gy1*cos + gy2*sin
        grad_x2 = -gy1*sin + gy2*cos

    Args:
        grad_out: Upstream gradient of shape (N, D) where N = B*T*H
        cos_vals: Cosine values of shape (N, D//2)
        sin_vals: Sine values of shape (N, D//2)

    Returns:
        grad_x: Gradient w.r.t. x, shape (N, D)
    """
    assert grad_out.ndim == 2, f"Expected 2D grad_out, got {grad_out.ndim}D"
    N, D = grad_out.shape
    half_d = D // 2
    assert cos_vals.shape == (N, half_d), f"cos shape mismatch: {cos_vals.shape}"
    assert sin_vals.shape == (N, half_d), f"sin shape mismatch: {sin_vals.shape}"

    total = N * D

    compiled = compile_kernel(
        _kernel_path("rotary_emb_backward.kernel"),
        "rotary_emb_backward"
    )
    kernel = compiled["kernel"]
    wgs = compiled["workgroup_size"]

    # params: vec4<u32> with N, D, half_d
    params = mx.array([N, D, half_d, 0], dtype=mx.uint32)

    grad_out_flat = grad_out.reshape(-1)
    cos_flat = cos_vals.reshape(-1)
    sin_flat = sin_vals.reshape(-1)

    outputs = kernel(
        inputs=[grad_out_flat, cos_flat, sin_flat, params],
        output_shapes=[(total,)],
        output_dtypes=[grad_out.dtype],
        grid=(_ceil_div(total, wgs[0]) * wgs[0], 1, 1),
        threadgroup=wgs,
        template=[("T", mx.float32)],
    )

    return outputs[0].reshape(N, D)


# ---------------------------------------------------------------------------
# Cross-Entropy Backward
# ---------------------------------------------------------------------------

def cross_entropy_backward(logits: mx.array, targets: mx.array) -> mx.array:
    """Backward pass for cross-entropy loss via Metal kernel.

    Computes grad_logits = softmax(logits) - one_hot(target)

    Args:
        logits: Raw logits of shape (B*T, V)
        targets: Integer target indices of shape (B*T,)

    Returns:
        grad_logits: Gradient w.r.t. logits, shape (B*T, V)
    """
    assert logits.ndim == 2, f"Expected 2D logits, got {logits.ndim}D"
    BT, V = logits.shape
    assert targets.shape == (BT,), f"Target shape mismatch: {targets.shape}"

    # Targets need to be float for the kernel (stored as f32, read as u32 inside)
    targets_f = targets.astype(mx.float32)

    compiled = compile_kernel(
        _kernel_path("cross_entropy_backward.kernel"),
        "cross_entropy_backward"
    )
    kernel = compiled["kernel"]
    wgs = compiled["workgroup_size"]

    # params: vec4<f32> with V (vocab size)
    params = mx.array([float(V), 0.0, 0.0, 0.0], dtype=mx.float32)

    outputs = kernel(
        inputs=[logits, targets_f, params],
        output_shapes=[logits.shape],
        output_dtypes=[logits.dtype],
        grid=(BT * wgs[0], 1, 1),
        threadgroup=wgs,
        template=[("T", mx.float32)],
    )

    return outputs[0]
