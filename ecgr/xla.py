"""Make XLA find CUDA's libdevice, or say clearly why it cannot.

Every training stage on this machine dies at its first step with

    error: libdevice not found at ./libdevice.10.bc
    UNKNOWN: JIT compilation failed. [[gradient_tape/.../ssm0_ssm/Sigmoid]]

and the reason is not in this project's code. TensorFlow's pip CUDA wheels
(`tensorflow[and-cuda]`) ship cuDNN and cuBLAS but not `nvvm/libdevice/libdevice.10.bc`,
which XLA needs to emit PTX. It is not avoidable by turning XLA off, either: neither
`jit_compile=False` on the model nor `TF_XLA_FLAGS=--tf_xla_auto_jit=0` prevents it, because
some fused activation gradients are lowered through XLA regardless of those settings.

So the file has to be found. `ensure_libdevice()` looks in the places it actually turns up -
a real CUDA toolkit, the `nvidia-cuda-nvcc` wheel, or Triton's bundled copy, which is what a
plain `pip install tensorflow[and-cuda] torch` leaves behind - and points XLA at it through
`XLA_FLAGS`. When the copy it finds is not laid out as `<dir>/nvvm/libdevice/`, it builds
that layout once in a cache directory out of symlinks.

Call it BEFORE TensorFlow compiles anything; cli.main() does, as the first thing it does.
"""
import os
import sys

LIBDEVICE = 'libdevice.10.bc'
_FLAG = '--xla_gpu_cuda_data_dir'


def _candidate_roots():
    """Directories that might hold `nvvm/libdevice/libdevice.10.bc`, best first."""
    roots = []
    for var in ('CUDA_HOME', 'CUDA_PATH', 'CUDA_ROOT'):
        if os.environ.get(var):
            roots.append(os.environ[var])
    roots += ['/usr/local/cuda', '/usr/lib/cuda', '/usr']
    for path in sys.path:
        if path.endswith('site-packages'):
            roots.append(os.path.join(path, 'nvidia', 'cuda_nvcc'))
    return roots


def _candidate_files():
    """Loose copies of libdevice.10.bc that are NOT in an nvvm/libdevice layout."""
    files = []
    for path in sys.path:
        if not path.endswith('site-packages'):
            continue
        files.append(os.path.join(path, 'triton', 'backends', 'nvidia', 'lib', LIBDEVICE))
        files.append(os.path.join(path, 'nvidia', 'cuda_nvcc', 'nvvm', 'libdevice',
                                  LIBDEVICE))
    return files


def _cache_dir():
    base = (os.environ.get('XDG_CACHE_HOME')
            or os.path.join(os.path.expanduser('~'), '.cache'))
    return os.path.join(base, 'ecgr', 'cuda')


def find_cuda_data_dir():
    """A directory D such that D/nvvm/libdevice/libdevice.10.bc exists, or None.

    Builds one out of symlinks in the cache directory when only a loose copy is available.
    """
    for root in _candidate_roots():
        if os.path.exists(os.path.join(root, 'nvvm', 'libdevice', LIBDEVICE)):
            return root

    for source in _candidate_files():
        if not os.path.exists(source):
            continue
        root = _cache_dir()
        target_dir = os.path.join(root, 'nvvm', 'libdevice')
        target = os.path.join(target_dir, LIBDEVICE)
        try:
            os.makedirs(target_dir, exist_ok=True)
            if not os.path.exists(target):
                os.symlink(os.path.realpath(source), target)
            return root
        except OSError:
            continue
    return None


def ensure_libdevice(verbose=True):
    """Add --xla_gpu_cuda_data_dir to XLA_FLAGS if it is not set and can be resolved.

    Returns the directory in use, or None. Never raises: a CPU-only machine has no need of
    this, and a warning is more useful than a failure at import time.
    """
    existing = os.environ.get('XLA_FLAGS', '')
    if _FLAG in existing:
        return None                              # the caller already chose one

    root = find_cuda_data_dir()
    if root is None:
        if verbose:
            print(f"warning: {LIBDEVICE} not found. If training fails with 'libdevice not "
                  f"found', install a CUDA toolkit or `pip install nvidia-cuda-nvcc-cu12` "
                  f"and set XLA_FLAGS={_FLAG}=<its root>.", file=sys.stderr)
        return None

    os.environ['XLA_FLAGS'] = f"{existing} {_FLAG}={root}".strip()
    if verbose:
        print(f"xla          : {_FLAG}={root}")
    return root
