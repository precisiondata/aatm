# Installation

The recommended way to install AATM is with [uv](https://docs.astral.sh/uv/), which
also picks the right PyTorch build for your hardware (CUDA, Apple GPU or CPU) in a
single command:

```bash
uv pip install aatm --torch-backend=auto
```

`--torch-backend` is a preview feature of uv. If it does not work for you, use pip as
shown below.

With pip:

```bash
pip install aatm
aatm doctor
```

`aatm doctor` checks whether local models can use your GPU and, if not, prints the
exact commands to fix it. See [GPU acceleration](#gpu-acceleration-optional) below.

If you want to build from the source, clone the repository and install it locally:

=== "pip"

    ```bash
    git clone https://github.com/precisiondata/aatm.git
    pip install -e . 
    ```

=== "uv"

    ```bash
    git clone https://github.com/precisiondata/aatm.git
    uv sync
    ```

## GPU acceleration (optional)

AATM does not require a GPU. Local models (embedding functions and rerankers) run on
the best device PyTorch can see: an NVIDIA GPU (CUDA), an Apple Silicon GPU (MPS), or
the CPU. API-based models (OpenAI, Google) never use a local GPU.

You only need the NVIDIA **driver**, not the CUDA Toolkit: PyTorch's CUDA builds ship
their own CUDA runtime.

### Check your setup with `aatm doctor`

```bash
aatm doctor
```

It reports your GPU, the highest CUDA version your driver supports (from
`nvidia-smi`), the installed PyTorch build and the device AATM will use. When the GPU
is not being used, it explains why and prints the commands that fix it, targeting the
Python environment AATM is installed in. It never changes anything itself. It exits
with code 1 when an action is needed, so it can be used in scripts.

### What `pip install aatm` gives you

PyTorch's default builds on PyPI differ by platform:

| Platform | Default PyTorch from PyPI | GPU support |
|---|---|---|
| Linux (x86_64) | CUDA build (currently CUDA 13) | Works if your driver supports that CUDA version |
| Windows | CPU-only build | Install a CUDA build to use an NVIDIA GPU (below) |
| macOS (Apple Silicon) | Build with MPS | Apple GPUs work out of the box |

On Linux, run `nvidia-smi`: the top-right corner shows the highest CUDA version your
driver supports. If it is older than the CUDA version of the default PyTorch build,
update the driver or install a matching build as shown below. `aatm doctor` does this
check for you.

### Installing a CUDA build of PyTorch manually

`aatm doctor` prints these commands with the right CUDA version filled in. To do it
by hand, pick the CUDA version your driver supports (see `nvidia-smi`, or the
[PyTorch install selector](https://pytorch.org/get-started/locally/)).

**Before installing AATM**, install PyTorch first. pip then keeps that build:

=== "pip"

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install aatm
    ```

=== "uv"

    ```bash
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install aatm
    ```

**If AATM is already installed**, torch is already present, so a plain
`pip install torch --index-url ...` does nothing. Replace it instead:

=== "pip"

    ```bash
    pip uninstall -y torch
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    ```

=== "uv"

    ```bash
    uv pip install --reinstall-package torch torch --index-url https://download.pytorch.org/whl/cu128
    ```

On Linux machines without a GPU, you can avoid downloading several GB of CUDA libraries
by installing the CPU-only build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install aatm
```

### Choosing the device

When a local model loads, AATM logs the device it runs on. When it falls back to CPU,
the log says why and suggests `aatm doctor`. Set `AATM_DEVICE` to `cuda`, `mps` or
`cpu` to override the automatic choice (see [Environment variables](env.md)).

### Working from a clone of the repository

`uv sync` installs the CUDA 12.8 build of PyTorch on Linux and Windows. On a machine
without an NVIDIA GPU, use the CPU-only build instead:

```bash
uv sync --no-group cu128 --group cpu
```
