# Phase 7: Deployment - Research

**Researched:** 2026-01-30
**Domain:** Model Export (ONNX), Containerization (Docker+CUDA), REST API (FastAPI), Distribution (PyPI/GitHub)
**Confidence:** HIGH

## Summary

This phase covers packaging the trained SAR autoencoder models for production deployment. The research focused on four key areas: ONNX export with preprocessing embedded in the graph, Docker containerization with NVIDIA CUDA support, FastAPI for REST serving, and Python package distribution via PyPI and GitHub Releases.

The standard approach for PyTorch 2.x model export is using `torch.onnx.export(..., dynamo=True)` which leverages TorchFX for graph capture. This is the recommended exporter as of PyTorch 2.5+ (and default as of 2.9). For GPU inference, ONNX Runtime with CUDAExecutionProvider provides seamless integration with PyTorch's CUDA environment. Docker containers should use NVIDIA's official `nvidia/cuda:*-runtime-*` base images for production, with multi-stage builds to minimize image size. FastAPI with Uvicorn is the standard for async REST APIs with file upload support.

**Primary recommendation:** Export all three ResNet variants (4x, 8x, 16x) to ONNX with embedded preprocessing, use a multi-stage Docker build with `nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04` base, and implement FastAPI endpoints with streaming file uploads for large GeoTIFF files.

## Standard Stack

The established libraries/tools for this domain:

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch.onnx | PyTorch 2.5+ | ONNX export | Built-in, dynamo-based exporter is current standard |
| onnxruntime-gpu | 1.19+ | ONNX inference | Official ONNX Runtime with CUDA 12.x support |
| fastapi | 0.115+ | REST API framework | Async, automatic OpenAPI docs, file uploads |
| uvicorn | 0.34+ | ASGI server | Production-grade async server |
| python-multipart | 0.0.18+ | File uploads | Required for FastAPI multipart form-data |
| rasterio | 1.3+ | GeoTIFF I/O | Already used in project, supports MemoryFile |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| aiofiles | 24.1+ | Async file I/O | Large file streaming to disk |
| onnx | 1.17+ | ONNX model inspection | Validation and model surgery |
| onnxruntime-tools | 1.7+ | FP16 conversion | Optional FP16 optimization |
| build | 1.2+ | Package building | PyPI distribution |
| twine | 6.0+ | PyPI upload | Package publishing |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| ONNX Runtime | TensorRT | TensorRT is faster but NVIDIA-only, more complex setup |
| FastAPI | Flask | Flask is simpler but lacks async, auto-docs, type hints |
| nvidia/cuda base | pytorch/pytorch | PyTorch images are larger, include dev tools |
| FP32 models | FP16 models | FP16 is 50% smaller/faster but may have precision loss |

**Installation:**
```bash
# Export dependencies
pip install onnx onnxruntime-gpu

# API dependencies
pip install fastapi uvicorn[standard] python-multipart aiofiles

# Packaging dependencies
pip install build twine
```

## Architecture Patterns

### Recommended Project Structure
```
sarcodec/                      # Package root (for PyPI)
  src/
    sarcodec/
      __init__.py              # Package entry point, version
      cli.py                   # CLI implementation
      api/
        __init__.py
        app.py                 # FastAPI application
        endpoints.py           # /compress, /decompress, etc.
        models.py              # Pydantic request/response models
      inference/
        onnx_runtime.py        # ONNX Runtime wrapper
        compressor.py          # Existing SARCompressor (keep)
      export/
        onnx_export.py         # ONNX export utilities
  scripts/
    export_onnx.py             # Export script (calls export module)
  Dockerfile                   # Multi-stage Docker build
  docker-compose.yml           # Local development setup
  pyproject.toml               # Package metadata
  docs/
    deployment.md              # Deployment guide
```

### Pattern 1: Wrapper Model for End-to-End ONNX Export

**What:** Embed preprocessing (dB conversion, normalization) inside the model for self-contained ONNX graphs.
**When to use:** Always - ensures ONNX model accepts raw linear SAR values.
**Example:**
```python
# Source: PyTorch ONNX Export tutorial + custom for this project
import torch
import torch.nn as nn

class SARAutoencoderWithPreprocess(nn.Module):
    """Wraps autoencoder with preprocessing for ONNX export."""

    def __init__(self, encoder, decoder, vmin: float, vmax: float):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # Register as buffers so they're exported
        self.register_buffer('vmin', torch.tensor(vmin))
        self.register_buffer('vmax', torch.tensor(vmax))
        self.register_buffer('noise_floor', torch.tensor(1e-10))

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert linear SAR to normalized [0,1] range."""
        # Clamp to noise floor
        x = torch.clamp(x, min=self.noise_floor)
        # Convert to dB
        x_db = 10.0 * torch.log10(x)
        # Clip to range
        x_db = torch.clamp(x_db, self.vmin, self.vmax)
        # Normalize to [0, 1]
        return (x_db - self.vmin) / (self.vmax - self.vmin)

    def inverse_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert normalized [0,1] back to linear SAR."""
        x_db = x * (self.vmax - self.vmin) + self.vmin
        return torch.pow(10.0, x_db / 10.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw SAR to latent."""
        return self.encoder(self.preprocess(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to raw SAR."""
        return self.inverse_preprocess(self.decoder(z))
```

### Pattern 2: Dynamic Batch Size ONNX Export

**What:** Export ONNX model with dynamic batch dimension for flexible inference.
**When to use:** Always - allows single and batch inference from same model.
**Example:**
```python
# Source: PyTorch 2.10 ONNX documentation
import torch
from torch.export import Dim

def export_with_dynamic_batch(model, example_input, output_path):
    """Export model with dynamic batch dimension."""

    # Define dynamic batch dimension
    batch_dim = Dim("batch_size", min=1, max=256)

    # Export with dynamo=True (recommended for PyTorch 2.5+)
    onnx_program = torch.onnx.export(
        model,
        example_input,
        dynamo=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"input": {0: batch_dim}},
        opset_version=20,
    )

    # Optimize and save
    onnx_program.optimize()
    onnx_program.save(output_path)

    return onnx_program
```

### Pattern 3: ONNX Runtime GPU Inference

**What:** Run ONNX model inference on GPU using CUDAExecutionProvider.
**When to use:** Always for GPU inference - faster than PyTorch for production.
**Example:**
```python
# Source: ONNX Runtime Python documentation
import onnxruntime as ort
import numpy as np

class ONNXInferenceSession:
    """ONNX Runtime inference wrapper with GPU support."""

    def __init__(self, model_path: str, device: str = "cuda"):
        # Select providers based on device
        if device == "cuda":
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                }),
                "CPUExecutionProvider",  # Fallback
            ]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def run(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data."""
        return self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )[0]
```

### Pattern 4: FastAPI File Upload with Streaming

**What:** Handle large GeoTIFF uploads efficiently using UploadFile streaming.
**When to use:** For the `/compress` and `/decompress` endpoints.
**Example:**
```python
# Source: FastAPI documentation
from fastapi import FastAPI, UploadFile, File, HTTPException
from rasterio.io import MemoryFile
import tempfile
import os

app = FastAPI()

@app.post("/compress")
async def compress_image(
    file: UploadFile = File(...),
    model: str = "8x"  # Default model
):
    """Compress a GeoTIFF image."""

    # Validate content type
    if not file.filename.endswith(('.tif', '.tiff')):
        raise HTTPException(400, "File must be a GeoTIFF (.tif/.tiff)")

    # Read file content
    content = await file.read()

    # Process with rasterio MemoryFile
    with MemoryFile(content) as memfile:
        with memfile.open() as src:
            data = src.read(1)  # Read first band
            geo_metadata = {
                "crs": src.crs.to_wkt() if src.crs else None,
                "transform": list(src.transform)[:6],
                "nodata": src.nodata,
            }

    # Compress (implementation depends on model)
    latent, metadata = await compress_data(data, model, geo_metadata)

    # Return compressed data as binary response
    return Response(
        content=serialize_compressed(latent, metadata),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={file.filename}.npz"}
    )
```

### Pattern 5: Docker Multi-Stage Build

**What:** Use multi-stage build to minimize production image size.
**When to use:** Always for Docker images.
**Example:**
```dockerfile
# Source: Docker and NVIDIA best practices
# Stage 1: Build stage (includes dev tools)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage (minimal)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python and minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application
COPY src/ ./src/
COPY models/ ./models/

# Entrypoint
ENTRYPOINT ["python", "-m", "sarcodec"]
CMD ["--help"]
```

### Anti-Patterns to Avoid

- **Hardcoding preprocessing values:** Always embed vmin/vmax in the ONNX model, don't require external config.
- **Using devel images in production:** Runtime images are 3-5x smaller; use multi-stage builds.
- **Loading entire file to memory:** For large GeoTIFFs (>1GB), use streaming or chunked processing.
- **Single worker in container:** Use Uvicorn's `--workers` flag for multi-core utilization (unless in Kubernetes).
- **Skipping ONNX validation:** Always verify numerical output matches PyTorch before deploying.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ONNX export | Custom serialization | torch.onnx.export(dynamo=True) | Handles all PyTorch ops, dynamic shapes, optimizations |
| GPU inference | PyTorch in production | onnxruntime-gpu | 20-30% faster, no Python GIL issues |
| File uploads | Custom multipart parser | FastAPI UploadFile | Handles streaming, temp files, async |
| GeoTIFF in-memory | Temp files | rasterio.MemoryFile | Avoids disk I/O, thread-safe |
| API documentation | Swagger manually | FastAPI auto-docs | Generated from type hints at /docs |
| FP16 conversion | Manual casting | onnxruntime-tools | Handles mixed precision properly |
| Docker GPU access | Manual CUDA install | nvidia-container-toolkit | Standard, maintained by NVIDIA |

**Key insight:** Model serving has well-established patterns. The innovation is in the model, not the serving infrastructure.

## Common Pitfalls

### Pitfall 1: ONNX Export with Batch Size 1

**What goes wrong:** Dynamic batch dimension not properly set when exporting with batch_size=1 example input.
**Why it happens:** PyTorch tracer may not infer the batch dimension is dynamic with single-element batch.
**How to avoid:** Use batch_size > 1 for example inputs (e.g., 2 or 4) when exporting with dynamic_shapes.
**Warning signs:** ONNX model only accepts the exact batch size used during export.

### Pitfall 2: Preprocessing Not in ONNX Graph

**What goes wrong:** ONNX model expects normalized input, but users provide raw SAR values.
**Why it happens:** Preprocessing done outside model.forward() is not captured.
**How to avoid:** Wrap model with preprocessing in a nn.Module, export the wrapper.
**Warning signs:** ONNX output values are wildly different from PyTorch (order of magnitude).

### Pitfall 3: FP16 Precision Loss

**What goes wrong:** Reconstructed images have visible artifacts or numerical instability.
**Why it happens:** Some layers (especially in decoder upsampling) lose precision with FP16.
**How to avoid:** Keep key layers in FP32 using mixed precision, or stick with FP32 for quality-critical models.
**Warning signs:** PSNR drops >0.5 dB after FP16 conversion, NaN values in output.

### Pitfall 4: Docker CUDA Version Mismatch

**What goes wrong:** Container fails with CUDA errors despite GPU being available.
**Why it happens:** Host driver version < container CUDA version, or mismatched cuDNN.
**How to avoid:** Use CUDA 12.1 base (widely supported), check driver compatibility matrix.
**Warning signs:** `CUDA driver version is insufficient`, `cuDNN version mismatch`.

### Pitfall 5: Large File Memory Exhaustion

**What goes wrong:** API crashes when processing large GeoTIFF (>1GB).
**Why it happens:** FastAPI's UploadFile loads to memory by default.
**How to avoid:** Stream to temp file first, then process; or use chunked processing.
**Warning signs:** OOMKilled in container logs, request timeouts on large files.

### Pitfall 6: Missing python-multipart

**What goes wrong:** FastAPI crashes on file upload with cryptic error.
**Why it happens:** python-multipart is not installed but required for multipart form-data.
**How to avoid:** Include in requirements.txt: `python-multipart>=0.0.18`.
**Warning signs:** `Error: python-multipart must be installed`.

### Pitfall 7: Docker Entrypoint Not Using Exec Form

**What goes wrong:** Container doesn't respond to SIGTERM, won't shut down gracefully.
**Why it happens:** Shell form CMD runs process as subprocess, signals don't propagate.
**How to avoid:** Use exec form: `CMD ["python", "-m", "sarcodec"]` not `CMD python -m sarcodec`.
**Warning signs:** Container takes 10+ seconds to stop, lifespan events not triggered.

## Code Examples

Verified patterns from official sources:

### ONNX Export with Validation

```python
# Source: PyTorch ONNX tutorial + ONNX Runtime docs
import torch
import onnxruntime as ort
import numpy as np

def export_and_validate(
    pytorch_model,
    example_input: torch.Tensor,
    output_path: str,
    atol: float = 1e-5
) -> bool:
    """Export model to ONNX and validate numerical accuracy."""

    pytorch_model.eval()

    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = pytorch_model(example_input)

    # Export to ONNX
    batch_dim = torch.export.Dim("batch", min=1, max=256)
    onnx_program = torch.onnx.export(
        pytorch_model,
        (example_input,),
        dynamo=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"input": {0: batch_dim}},
    )
    onnx_program.optimize()
    onnx_program.save(output_path)

    # Validate with ONNX Runtime
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"input": example_input.numpy()})[0]

    # Compare outputs
    max_diff = np.abs(pytorch_output.numpy() - onnx_output).max()
    print(f"Max difference: {max_diff:.2e}")

    if max_diff < atol:
        print("Validation PASSED")
        return True
    else:
        print(f"Validation FAILED (max_diff={max_diff:.2e} > atol={atol:.2e})")
        return False
```

### FastAPI Application Structure

```python
# Source: FastAPI documentation
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import io

# Global model storage
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    # Startup: load ONNX models
    for name in ["4x", "8x", "16x"]:
        models[name] = load_onnx_model(f"models/resnet_{name}.onnx")
    yield
    # Shutdown: cleanup
    models.clear()

app = FastAPI(
    title="SAR Codec API",
    description="SAR image compression using learned autoencoders",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": list(models.keys())}

@app.post("/encode")
async def encode(
    file: UploadFile = File(...),
    model: str = Query("8x", enum=["4x", "8x", "16x"]),
):
    """Encode a GeoTIFF to latent representation."""
    content = await file.read()
    latent, metadata = await process_encode(content, models[model])

    # Return as NPZ binary
    buffer = io.BytesIO()
    np.savez_compressed(buffer, latent=latent, metadata=metadata)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={file.filename}.npz"}
    )
```

### Docker Compose for Development

```yaml
# Source: Docker and NVIDIA best practices
version: "3.8"

services:
  sarcodec:
    build:
      context: .
      dockerfile: Dockerfile
    image: sarcodec:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./data:/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: ["--serve", "--host", "0.0.0.0", "--port", "8000"]
    shm_size: 1g  # Increase shared memory for PyTorch/NCCL
```

### pyproject.toml for PyPI Distribution

```toml
# Source: Python Packaging User Guide
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "sarcodec"
version = "1.0.0"
description = "SAR image compression using learned autoencoders"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.10"
authors = [
    {name = "Author Name", email = "author@example.com"}
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Image Processing",
]
dependencies = [
    "numpy>=1.24",
    "onnxruntime>=1.17",
    "rasterio>=1.3",
    "rich>=13.0",
]

[project.optional-dependencies]
gpu = ["onnxruntime-gpu>=1.19"]
api = ["fastapi>=0.115", "uvicorn[standard]>=0.34", "python-multipart>=0.0.18"]
dev = ["pytest", "pytest-cov", "black", "ruff"]

[project.scripts]
sarcodec = "sarcodec.cli:main"

[project.urls]
Homepage = "https://github.com/username/sarcodec"
Documentation = "https://github.com/username/sarcodec#readme"
Repository = "https://github.com/username/sarcodec"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| torch.onnx.export (TorchScript) | torch.onnx.export(dynamo=True) | PyTorch 2.5 (Oct 2024) | Better dynamic shape support, FX-based |
| onnxruntime CUDA 11.x | onnxruntime CUDA 12.x default | onnxruntime 1.19 (2024) | Better performance, newer GPU support |
| Gunicorn + Uvicorn | Uvicorn with --workers | Uvicorn 0.30+ | Simpler, sufficient for most cases |
| setup.py packaging | pyproject.toml | PEP 621 (2020), widely adopted 2024+ | Declarative, standardized |

**Deprecated/outdated:**
- **TorchScript-based ONNX export:** Legacy approach, dynamo=True is now default (PyTorch 2.9+)
- **docker run --runtime=nvidia:** Replaced by `--gpus` flag with nvidia-container-toolkit
- **tiangolo/uvicorn-gunicorn-fastapi-docker:** No longer needed, build from scratch with Uvicorn workers

## Open Questions

Things that couldn't be fully resolved:

1. **Full Compressor ONNX Export Feasibility**
   - What we know: Can export the autoencoder model (encode/decode)
   - What's unclear: Can tiling/blending logic be exported? ONNX may not support complex control flow
   - Recommendation: Export patch-level model only; implement tiling in Python/ONNX Runtime wrapper

2. **PyPI Package Name Availability**
   - What we know: "sarcodec" is likely available (not a common term)
   - What's unclear: Need to verify on PyPI before publishing
   - Recommendation: Check `pip index versions sarcodec` before finalizing

3. **Optimal FP16 Strategy**
   - What we know: FP16 gives ~50% size reduction, ~1.3x speedup
   - What's unclear: Quality impact on SAR reconstruction (model-specific)
   - Recommendation: Export both FP32 and FP16, benchmark PSNR difference, let user choose

4. **Container Models Bundling**
   - What we know: All 3 models (4x, 8x, 16x) total ~100-200MB
   - What's unclear: Bundle all or download on demand?
   - Recommendation: Bundle all for simplicity; image size is dominated by CUDA libs anyway

## Sources

### Primary (HIGH confidence)
- [PyTorch ONNX Documentation (2.10)](https://docs.pytorch.org/docs/stable/onnx.html) - Export API, dynamo=True details
- [PyTorch ONNX Tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html) - Complete export example
- [ONNX Runtime Python Docs](https://onnxruntime.ai/docs/get-started/with-python.html) - GPU inference setup
- [ONNX Runtime CUDA Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) - CUDA configuration
- [FastAPI Request Files](https://fastapi.tiangolo.com/tutorial/request-files/) - File upload patterns
- [FastAPI Docker Deployment](https://fastapi.tiangolo.com/deployment/docker/) - Production deployment
- [Python Packaging Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) - pyproject.toml spec
- [NVIDIA CUDA Docker Images](https://hub.docker.com/r/nvidia/cuda) - Base image selection

### Secondary (MEDIUM confidence)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html) - GPU container setup
- [ONNX Float16 Optimization](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html) - FP16 conversion

### Tertiary (LOW confidence)
- GitHub Issues on dynamic shapes (PyTorch #126607, #148886) - Known issues with batch_size=1

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Official documentation verified, project already uses PyTorch 2.5+
- Architecture patterns: HIGH - Based on official tutorials and established practices
- ONNX export: HIGH - torch.onnx.export(dynamo=True) is well-documented
- Docker/CUDA: MEDIUM - CUDA 12.1 widely supported but version compatibility varies
- FastAPI patterns: HIGH - Official FastAPI documentation is comprehensive
- FP16 optimization: MEDIUM - Quality impact is model-specific, needs testing
- Pitfalls: HIGH - Based on documented issues and official warnings

**Research date:** 2026-01-30
**Valid until:** 2026-03-01 (30 days - stable domain, but check for PyTorch/ONNX updates)
