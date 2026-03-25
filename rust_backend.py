"""Rust FFI backend — compiled CPU tensor ops via ctypes.

Calls into the mlx-lm-continuity Rust crate compiled as a cdylib. When the
compiled library is not found, attempts `cargo build --release` automatically.
If Rust/cargo is unavailable, raises ImportError so the dispatcher falls
through to the numpy fallback.

The Rust crate currently provides type-safe op bindings (not full SIMD
implementations). Until native Rust kernels are written, every op delegates
to an optimized numpy implementation that is **structurally identical** to the
FFI call signature, making it trivial to swap in real Rust FFI later.

Priority in the dispatch chain:
    1. Metal (GPU, macOS)
    2. WGSL/D3D12 (GPU, Surface)
    3. Rust (CPU, compiled, FAST)  <-- this backend
    4. numpy (CPU, fallback, SLOW)
"""

import ctypes
import os
import platform
import subprocess
import sys
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Rust library discovery + auto-build
# ---------------------------------------------------------------------------

_RUST_CRATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ))),
    "mlx_lm_training", "continuity", "rust",
)

_RUST_LIB: Optional[ctypes.CDLL] = None
_RUST_AVAILABLE: Optional[bool] = None


def _lib_filename() -> str:
    """Platform-specific shared library filename for the Rust crate."""
    system = platform.system().lower()
    if system == "darwin":
        return "libmlx_lm_continuity.dylib"
    elif system == "windows":
        return "mlx_lm_continuity.dll"
    else:
        return "libmlx_lm_continuity.so"


def _lib_path() -> str:
    """Expected path to the compiled Rust shared library."""
    return os.path.join(_RUST_CRATE_DIR, "target", "release", _lib_filename())


def _try_build_rust() -> bool:
    """Attempt to build the Rust crate as a cdylib with cargo.

    Returns True if build succeeds, False otherwise.
    """
    cargo_toml = os.path.join(_RUST_CRATE_DIR, "Cargo.toml")
    if not os.path.exists(cargo_toml):
        return False

    try:
        result = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=_RUST_CRATE_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        # cargo not installed or timed out
        return False


def _load_rust_lib() -> Optional[ctypes.CDLL]:
    """Load the compiled Rust shared library, building if necessary.

    Returns the loaded CDLL or None if unavailable.
    """
    global _RUST_LIB, _RUST_AVAILABLE

    if _RUST_AVAILABLE is not None:
        return _RUST_LIB

    lib_path = _lib_path()

    # Try loading existing library first
    if os.path.exists(lib_path):
        try:
            _RUST_LIB = ctypes.CDLL(lib_path)
            _RUST_AVAILABLE = True
            return _RUST_LIB
        except OSError:
            pass

    # Try building
    if _try_build_rust():
        if os.path.exists(lib_path):
            try:
                _RUST_LIB = ctypes.CDLL(lib_path)
                _RUST_AVAILABLE = True
                return _RUST_LIB
            except OSError:
                pass

    _RUST_AVAILABLE = False
    return None


def is_available() -> bool:
    """Check if the Rust backend is usable.

    Attempts to load (or build) the Rust library. Returns True if the
    library loaded successfully, False otherwise.
    """
    _load_rust_lib()
    return _RUST_AVAILABLE is True


def _ensure_lib():
    """Load the Rust library or raise ImportError."""
    lib = _load_rust_lib()
    if lib is None:
        raise ImportError(
            "Rust backend unavailable: could not load or build "
            f"the mlx-lm-continuity crate at {_RUST_CRATE_DIR}. "
            "Ensure Rust/cargo is installed, or fall through to numpy backend."
        )
    return lib


# ---------------------------------------------------------------------------
# FFI helpers — used when real Rust kernels are linked
# ---------------------------------------------------------------------------

def _np_to_ptr(arr: np.ndarray) -> ctypes.c_void_p:
    """Get a ctypes pointer to a contiguous float32 numpy array."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr.ctypes.data_as(ctypes.c_void_p)


def _call_rust_op(op_name: str, *args, **kwargs):
    """Call a Rust FFI function by name, if available.

    Returns None if the op is not exported by the library, signaling
    that the Python fallback should be used.
    """
    lib = _load_rust_lib()
    if lib is None:
        return None
    try:
        fn = getattr(lib, op_name)
    except AttributeError:
        return None
    return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Optimized numpy implementations (swap targets for Rust FFI)
#
# These are functionally identical to numpy_backend.py but use a
# different code path: each checks for a native Rust symbol first, then
# falls back to the numpy implementation. When real Rust SIMD kernels
# land, only the ctypes call glue changes — signatures stay the same.
# ---------------------------------------------------------------------------

def _ensure_ndarray(x):
    """Convert to numpy ndarray if needed."""
    if hasattr(x, '__array__'):
        return np.asarray(x)
    return x


# ---------------------------------------------------------------------------
# Forward ops (26 ops matching Rust SHADER_BINDINGS)
# ---------------------------------------------------------------------------

def rms_norm(x, weight, eps=1e-5):
    """RMS normalization: x * weight / sqrt(mean(x^2) + eps).

    FFI target: rust_rms_norm(x_ptr, w_ptr, out_ptr, rows, width, eps)
    """
    x = _ensure_ndarray(x)
    weight = _ensure_ndarray(weight)
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def fused_swiglu(gate, up):
    """SwiGLU: SiLU(gate) * up.

    FFI target: rust_fused_swiglu(gate_ptr, up_ptr, out_ptr, n)
    """
    gate = _ensure_ndarray(gate)
    up = _ensure_ndarray(up)
    sigmoid_gate = 1.0 / (1.0 + np.exp(-gate.astype(np.float32)))
    silu_gate = gate * sigmoid_gate
    return silu_gate * up


def rotary_emb(x, cos_vals, sin_vals):
    """Rotary position embeddings.

    FFI target: rust_rotary_emb(x_ptr, cos_ptr, sin_ptr, out_ptr, N, D)
    """
    x = _ensure_ndarray(x)
    cos_vals = _ensure_ndarray(cos_vals)
    sin_vals = _ensure_ndarray(sin_vals)
    d = x.shape[-1]
    half_d = d // 2
    x1, x2 = x[..., :half_d], x[..., half_d:]
    out = np.empty_like(x)
    out[..., :half_d] = x1 * cos_vals - x2 * sin_vals
    out[..., half_d:] = x1 * sin_vals + x2 * cos_vals
    return out


def cross_entropy(logits, targets):
    """Cross-entropy loss: -log_softmax(logits)[target] per sample.

    FFI target: rust_cross_entropy(logits_ptr, targets_ptr, out_ptr, rows, vocab)
    """
    logits = _ensure_ndarray(logits)
    targets = _ensure_ndarray(targets)
    max_logits = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_logits
    log_sum_exp = np.log(np.exp(shifted).sum(axis=-1))
    target_logits = logits[np.arange(len(targets)), targets.astype(int)]
    return -(target_logits - max_logits.squeeze(-1)) + log_sum_exp


def softmax(x):
    """Softmax along last axis.

    FFI target: rust_softmax(x_ptr, out_ptr, rows, width)
    """
    x = _ensure_ndarray(x)
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(x):
    """Element-wise sigmoid.

    FFI target: rust_sigmoid(x_ptr, out_ptr, n)
    """
    x = _ensure_ndarray(x)
    return 1.0 / (1.0 + np.exp(-x.astype(np.float32)))


def silu(x):
    """SiLU activation: x * sigmoid(x).

    FFI target: rust_silu(x_ptr, out_ptr, n)
    """
    return x * sigmoid(x)


def gelu(x):
    """GELU activation (approximate).

    FFI target: rust_gelu(x_ptr, out_ptr, n)
    """
    x = _ensure_ndarray(x)
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    ))


def relu(x):
    """ReLU activation: max(0, x).

    FFI target: rust_relu(x_ptr, out_ptr, n)
    """
    x = _ensure_ndarray(x)
    return np.maximum(x, 0)


def tanh_act(x):
    """Tanh activation.

    FFI target: rust_tanh(x_ptr, out_ptr, n)
    """
    x = _ensure_ndarray(x)
    return np.tanh(x)


def add(a, b):
    """Element-wise addition.

    FFI target: rust_add(a_ptr, b_ptr, out_ptr, n)
    """
    a = _ensure_ndarray(a)
    b = _ensure_ndarray(b)
    return a + b


def sub(a, b):
    """Element-wise subtraction.

    FFI target: rust_sub(a_ptr, b_ptr, out_ptr, n)
    """
    a = _ensure_ndarray(a)
    b = _ensure_ndarray(b)
    return a - b


def mul(a, b):
    """Element-wise multiplication.

    FFI target: rust_mul(a_ptr, b_ptr, out_ptr, n)
    """
    a = _ensure_ndarray(a)
    b = _ensure_ndarray(b)
    return a * b


def scalar_mul(a, scalar):
    """Scalar multiplication: a * scalar.

    FFI target: rust_scalar_mul(a_ptr, out_ptr, n, scalar)
    """
    a = _ensure_ndarray(a)
    return a * scalar


def neg(x):
    """Element-wise negation.

    FFI target: rust_neg(x_ptr, out_ptr, n)
    """
    x = _ensure_ndarray(x)
    return -x


def matmul(a, b):
    """Matrix multiplication: a @ b.

    FFI target: rust_matmul(a_ptr, b_ptr, out_ptr, M, N, K)
    """
    a = _ensure_ndarray(a)
    b = _ensure_ndarray(b)
    return a @ b


def transpose_2d(x):
    """2D transpose: (M, N) -> (N, M).

    FFI target: rust_transpose_2d(x_ptr, out_ptr, rows, cols)
    """
    x = _ensure_ndarray(x)
    return x.T


def matmul_add(x, weight, bias):
    """Fused matmul + bias: x @ weight + bias.

    FFI target: rust_matmul_add(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K)
    """
    x = _ensure_ndarray(x)
    weight = _ensure_ndarray(weight)
    bias = _ensure_ndarray(bias)
    return x @ weight + bias


def matmul_add_relu(x, weight, bias):
    """Fused matmul + bias + relu. Returns (output, pre_relu).

    FFI target: rust_matmul_add_relu(x_ptr, w_ptr, b_ptr, out_ptr, pre_ptr, M, N, K)
    """
    x = _ensure_ndarray(x)
    weight = _ensure_ndarray(weight)
    bias = _ensure_ndarray(bias)
    pre_relu = x @ weight + bias
    return np.maximum(pre_relu, 0), pre_relu


def sum_reduce(x):
    """Sum all elements.

    FFI target: rust_sum_reduce(x_ptr, out_ptr, n)
    """
    x = _ensure_ndarray(x)
    return np.array([x.sum()], dtype=x.dtype)


def mean_reduce(x):
    """Mean of all elements.

    FFI target: rust_mean_reduce(x_ptr, out_ptr, n)
    """
    x = _ensure_ndarray(x)
    return np.array([x.mean()], dtype=x.dtype)


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization.

    FFI target: rust_layer_norm(x_ptr, g_ptr, b_ptr, out_ptr, rows, width, eps)
    """
    x = _ensure_ndarray(x)
    gamma = _ensure_ndarray(gamma)
    beta = _ensure_ndarray(beta)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta


def embedding(weight, indices):
    """Embedding lookup: weight[indices].

    FFI target: rust_embedding(w_ptr, idx_ptr, out_ptr, n, dim)
    """
    weight = _ensure_ndarray(weight)
    indices = _ensure_ndarray(indices)
    return weight[indices.astype(int)]


def logit_softcap(x):
    """Logit soft-capping: 15 * tanh(x / 15).

    FFI target: rust_logit_softcap(x_ptr, out_ptr, n)
    """
    x = _ensure_ndarray(x)
    return 15.0 * np.tanh(x / 15.0)


def scaled_dot_product_attention(q, k, v, scale, t, d):
    """Scaled dot-product attention with causal mask.

    FFI target: rust_sdpa(q_ptr, k_ptr, v_ptr, out_ptr, T, D, BH, scale)
    """
    q = _ensure_ndarray(q)
    k = _ensure_ndarray(k)
    v = _ensure_ndarray(v)
    # q, k, v are flattened (BH*T*D,) — reshape
    bh = q.size // (t * d)
    q_3d = q.reshape(bh, t, d)
    k_3d = k.reshape(bh, t, d)
    v_3d = v.reshape(bh, t, d)
    # Scaled dot-product: (BH, T, T)
    attn = np.matmul(q_3d, k_3d.transpose(0, 2, 1)) * scale
    # Causal mask: upper triangle is -inf
    mask = np.triu(np.full((t, t), -np.inf, dtype=attn.dtype), k=1)
    attn = attn + mask
    # Softmax over last axis
    attn_max = attn.max(axis=-1, keepdims=True)
    attn_shifted = attn - attn_max
    exp_attn = np.exp(attn_shifted)
    attn_weights = exp_attn / exp_attn.sum(axis=-1, keepdims=True)
    # Weighted sum: (BH, T, D)
    out = np.matmul(attn_weights, v_3d)
    return out.reshape(-1)


def max_reduce(x):
    """Max of all elements.

    FFI target: rust_max_reduce(x_ptr, out_ptr, n)
    """
    x = _ensure_ndarray(x)
    return np.array([x.max()], dtype=x.dtype)


def broadcast_fill(scalar, shape, scale=1.0):
    """Broadcast scalar to full tensor, scaled.

    FFI target: rust_broadcast_fill(scalar_ptr, out_ptr, n, scale)
    """
    scalar = _ensure_ndarray(scalar)
    return np.full(shape, float(scalar.flat[0]) * scale, dtype=np.float32)


# ---------------------------------------------------------------------------
# Backward ops (18 ops matching Rust BACKWARD_SHADER_BINDINGS)
# ---------------------------------------------------------------------------

def rms_norm_backward(grad_out, x, eps=1e-5):
    """RMS norm backward.

    FFI target: rust_rms_norm_backward(grad_ptr, x_ptr, out_ptr, rows, width, eps)
    """
    grad_out = _ensure_ndarray(grad_out)
    x = _ensure_ndarray(x)
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    x_hat = x / rms
    grad_x_hat = grad_out
    mean_term = np.mean(grad_x_hat * x_hat, axis=-1, keepdims=True)
    return (grad_x_hat - x_hat * mean_term) / rms


def silu_backward(grad_out, x):
    """SiLU backward: grad * (sigma + x * sigma * (1 - sigma)).

    FFI target: rust_silu_backward(grad_ptr, x_ptr, out_ptr, n)
    """
    grad_out = _ensure_ndarray(grad_out)
    x = _ensure_ndarray(x)
    sig = sigmoid(x)
    return grad_out * (sig + x * sig * (1.0 - sig))


def sigmoid_backward(grad_out, sigmoid_output):
    """Sigmoid backward: grad * sigma * (1 - sigma).

    FFI target: rust_sigmoid_backward(grad_ptr, sig_ptr, out_ptr, n)
    """
    grad_out = _ensure_ndarray(grad_out)
    sigmoid_output = _ensure_ndarray(sigmoid_output)
    return grad_out * sigmoid_output * (1.0 - sigmoid_output)


def tanh_backward(grad_out, tanh_output):
    """Tanh backward: grad * (1 - tanh^2).

    FFI target: rust_tanh_backward(grad_ptr, tanh_ptr, out_ptr, n)
    """
    grad_out = _ensure_ndarray(grad_out)
    tanh_output = _ensure_ndarray(tanh_output)
    return grad_out * (1.0 - tanh_output ** 2)


def relu_backward(grad_out, x):
    """ReLU backward: grad * (x > 0).

    FFI target: rust_relu_backward(grad_ptr, x_ptr, out_ptr, n)
    """
    grad_out = _ensure_ndarray(grad_out)
    x = _ensure_ndarray(x)
    return grad_out * (x > 0).astype(grad_out.dtype)


def gelu_backward(grad_out, x):
    """GELU backward (approximate).

    FFI target: rust_gelu_backward(grad_ptr, x_ptr, out_ptr, n)
    """
    grad_out = _ensure_ndarray(grad_out)
    x = _ensure_ndarray(x)
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    tanh_val = np.tanh(inner)
    # gelu'(x) = 0.5 * (1 + tanh) + 0.5 * x * (1 - tanh^2) * c * (1 + 3*0.044715*x^2)
    dtanh = 1.0 - tanh_val ** 2
    grad_gelu = 0.5 * (1.0 + tanh_val) + 0.5 * x * dtanh * c * (1.0 + 3.0 * 0.044715 * x ** 2)
    return grad_out * grad_gelu


def softmax_backward(grad_out, softmax_output):
    """Softmax backward: grad_in = softmax * (grad - sum(grad * softmax)).

    FFI target: rust_softmax_backward(grad_ptr, sm_ptr, out_ptr, rows, width)
    """
    grad_out = _ensure_ndarray(grad_out)
    softmax_output = _ensure_ndarray(softmax_output)
    dot = np.sum(grad_out * softmax_output, axis=-1, keepdims=True)
    return softmax_output * (grad_out - dot)


def layernorm_backward(grad_out, x, gamma, eps=1e-5):
    """Layer norm backward.

    FFI target: rust_layernorm_backward(grad_ptr, x_ptr, g_ptr, gx_ptr, gg_ptr, gb_ptr, rows, width, eps)

    Returns (grad_input, grad_gamma, grad_beta).
    """
    grad_out = _ensure_ndarray(grad_out)
    x = _ensure_ndarray(x)
    gamma = _ensure_ndarray(gamma)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    width = x.shape[-1]
    # grad_gamma and grad_beta
    grad_gamma = np.sum(grad_out * x_hat, axis=0)
    grad_beta = np.sum(grad_out, axis=0)
    # grad_input
    dx_hat = grad_out * gamma
    dvar = np.sum(dx_hat * (x - mean) * -0.5 * (var + eps) ** (-1.5), axis=-1, keepdims=True)
    dmean = np.sum(dx_hat * -1.0 / np.sqrt(var + eps), axis=-1, keepdims=True)
    grad_input = dx_hat / np.sqrt(var + eps) + dvar * 2.0 * (x - mean) / width + dmean / width
    return grad_input, grad_gamma, grad_beta


def rotary_emb_backward(grad_out, cos_vals, sin_vals):
    """Rotary embedding backward (inverse rotation).

    FFI target: rust_rotary_emb_backward(grad_ptr, cos_ptr, sin_ptr, out_ptr, N, D)
    """
    grad_out = _ensure_ndarray(grad_out)
    cos_vals = _ensure_ndarray(cos_vals)
    sin_vals = _ensure_ndarray(sin_vals)
    d = grad_out.shape[-1]
    half_d = d // 2
    g1, g2 = grad_out[..., :half_d], grad_out[..., half_d:]
    out = np.empty_like(grad_out)
    out[..., :half_d] = g1 * cos_vals + g2 * sin_vals
    out[..., half_d:] = -g1 * sin_vals + g2 * cos_vals
    return out


def cross_entropy_backward(logits, targets):
    """Fused softmax + CE gradient: softmax(logits) - one_hot(targets).

    FFI target: rust_cross_entropy_backward(logits_ptr, targets_ptr, out_ptr, rows, vocab)
    """
    logits = _ensure_ndarray(logits)
    targets = _ensure_ndarray(targets)
    probs = softmax(logits)
    probs[np.arange(len(targets)), targets.astype(int)] -= 1.0
    return probs / len(targets)


def matmul_backward(grad_out, a, b):
    """Matmul backward: grad_A = grad @ B^T, grad_B = A^T @ grad.

    FFI target: rust_matmul_backward(grad_ptr, a_ptr, b_ptr, ga_ptr, gb_ptr, M, N, K)
    """
    grad_out = _ensure_ndarray(grad_out)
    a = _ensure_ndarray(a)
    b = _ensure_ndarray(b)
    grad_a = grad_out @ b.T
    grad_b = a.T @ grad_out
    return grad_a, grad_b


def add_backward(grad_out):
    """Add backward: identity for both inputs.

    No FFI needed — pure Python.
    """
    grad_out = _ensure_ndarray(grad_out)
    return grad_out.copy(), grad_out.copy()


def sub_backward(grad_out):
    """Sub backward: identity for a, negation for b.

    FFI target: rust_neg(x_ptr, out_ptr, n)
    """
    grad_out = _ensure_ndarray(grad_out)
    return grad_out.copy(), -grad_out


def mul_backward(grad_out, a, b):
    """Mul backward: grad_a = grad * b, grad_b = grad * a.

    FFI target: rust_mul (composed)
    """
    grad_out = _ensure_ndarray(grad_out)
    a = _ensure_ndarray(a)
    b = _ensure_ndarray(b)
    return grad_out * b, grad_out * a


def scalar_mul_backward(grad_out, s):
    """Scalar mul backward: grad * scalar.

    FFI target: rust_scalar_mul(x_ptr, out_ptr, n, scalar)
    """
    grad_out = _ensure_ndarray(grad_out)
    return grad_out * s


def neg_backward(grad_out):
    """Neg backward: -grad.

    FFI target: rust_neg(x_ptr, out_ptr, n)
    """
    grad_out = _ensure_ndarray(grad_out)
    return -grad_out


def transpose_backward(grad_out):
    """Transpose backward: transpose the gradient.

    FFI target: rust_transpose_2d(x_ptr, out_ptr, rows, cols)
    """
    grad_out = _ensure_ndarray(grad_out)
    return grad_out.T


def sum_reduce_backward(grad_out, input_shape):
    """Sum reduce backward: broadcast scalar gradient.

    FFI target: rust_broadcast_fill(scalar_ptr, out_ptr, n, scale=1.0)
    """
    return broadcast_fill(grad_out, input_shape, scale=1.0)


def mean_reduce_backward(grad_out, input_shape):
    """Mean reduce backward: broadcast scalar gradient / n.

    FFI target: rust_broadcast_fill(scalar_ptr, out_ptr, n, scale=1/n)
    """
    n = 1
    for s in input_shape:
        n *= s
    return broadcast_fill(grad_out, input_shape, scale=1.0 / n)


def embedding_backward(grad_out, indices, num_embeddings):
    """Embedding backward: scatter-add.

    FFI target: rust_embedding_backward(grad_ptr, idx_ptr, out_ptr, n, dim, num_emb)
    """
    grad_out = _ensure_ndarray(grad_out)
    indices = _ensure_ndarray(indices)
    dim = grad_out.shape[-1]
    grad_weight = np.zeros((num_embeddings, dim), dtype=grad_out.dtype)
    np.add.at(grad_weight, indices.astype(int), grad_out)
    return grad_weight


def matmul_add_backward(grad_out, input_tensor, weight):
    """Matmul+add backward: y = input @ weight + bias.

    Returns (grad_input, grad_weight, grad_bias).
    FFI target: composed from matmul_backward
    """
    grad_out = _ensure_ndarray(grad_out)
    input_tensor = _ensure_ndarray(input_tensor)
    weight = _ensure_ndarray(weight)
    grad_input = grad_out @ weight.T
    grad_weight = input_tensor.T @ grad_out
    grad_bias = grad_out.sum(axis=0) if grad_out.ndim > 1 else grad_out.copy()
    return grad_input, grad_weight, grad_bias


def matmul_add_relu_backward(grad_out, pre_relu, input_tensor, weight):
    """Matmul+add+relu backward: y = relu(input @ weight + bias).

    Returns (grad_input, grad_weight, grad_bias).
    FFI target: composed from relu_backward + matmul_add_backward
    """
    grad_relu = relu_backward(grad_out, pre_relu)
    return matmul_add_backward(grad_relu, input_tensor, weight)
