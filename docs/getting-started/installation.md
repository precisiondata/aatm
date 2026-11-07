# Installation

Install the package in your environment.

```bash
pip install aatm
```

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

What `pip install aatm` gives you depends on your platform, because PyTorch's
default builds on PyPI differ:

| Platform | Default PyTorch from PyPI | GPU support |
|---|---|---|
| Linux (x86_64) | CUDA build | NVIDIA GPUs work out of the box, given a recent driver |
| Windows | CPU-only build | Install a CUDA build to use an NVIDIA GPU (below) |
| macOS (Apple Silicon) | Build with MPS | Apple GPUs work out of the box |

### Installing a CUDA build of PyTorch

If you have an NVIDIA GPU and your PyTorch build has no CUDA support (for example on
Windows), install PyTorch first using the
[PyTorch install selector](https://pytorch.org/get-started/locally/), then install AATM.
Pick the CUDA version your driver supports (`nvidia-smi` shows the highest one):

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

With uv you can also let it detect your GPU driver and choose a matching PyTorch build.
This uv feature is in preview:

```bash
uv pip install aatm --torch-backend=auto
```

On Linux machines without a GPU, you can avoid downloading the large CUDA libraries by
installing the CPU-only build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install aatm
```

### Checking which device is used

When a local model loads, AATM logs the device it runs on, and a hint when it falls
back to CPU. You can also check directly:

```bash
python -c "from aatm.device import get_device; print(get_device())"
```

Set `AATM_DEVICE` to `cuda`, `mps` or `cpu` to override the automatic choice (see
[Environment variables](env.md)).

### Working from a clone of the repository

`uv sync` installs the CUDA 12.8 build of PyTorch on Linux and Windows. On a machine
without an NVIDIA GPU, use the CPU-only build instead:

```bash
uv sync --no-group cu128 --group cpu
```
