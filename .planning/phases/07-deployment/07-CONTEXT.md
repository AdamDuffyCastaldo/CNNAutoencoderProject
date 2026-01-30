# Phase 7: Deployment - Context

**Gathered:** 2026-01-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Package the best-performing ResNet autoencoder models for production deployment. Export to ONNX format, containerize with Docker (GPU), create REST API, and distribute via GitHub Releases and PyPI.

</domain>

<decisions>
## Implementation Decisions

### Export Formats
- **Primary format:** ONNX (not TorchScript)
- **Models to export:** All 3 ResNet variants (4x, 8x, 16x) - user selects quality/size tradeoff
- **Preprocessing:** End-to-end - include normalization in ONNX graph
- **Batch size:** Dynamic - support variable batch sizes
- **Scope:** Attempt to export full compressor with tiled inference (not just patch model)
- **Validation:** Both numerical (threshold check) and visual (comparison images)

### Container Strategy
- **GPU support:** Required - NVIDIA CUDA base image
- **Entry point:** Both modes - CLI by default, `--serve` flag starts API
- **Docker Compose:** Yes - include for easy local setup
- **Health checks:** No - keep container simple

### API Design
- **Framework:** FastAPI
- **File upload:** Multipart form-data (standard approach)
- **Authentication:** None - open API for local/internal use
- **Endpoints:** Full pipeline - `/compress`, `/decompress`, `/encode`, `/decode`

### Distribution
- **Primary channel:** GitHub Releases (ONNX models + scripts as assets)
- **PyPI package:** Yes - `pip install` for easy Python integration
- **Documentation:** `docs/` folder with separate guides
- **Versioning:** Semantic versioning (1.0.0)
- **Examples:** Include Jupyter notebooks showing usage
- **License:** Apache 2.0

### Claude's Discretion
- Base image selection (optimize for size vs compatibility)
- ONNX optimization level (standard vs ONNX Runtime optimized)
- Model quantization (FP32 vs FP16 - assess quality impact)
- Models bundled in container (all vs default only)
- Processing mode for API (sync vs async polling)
- Response format for compressed data (binary vs JSON)
- Model selection API style (query param vs path)
- PyPI package name (check availability)

</decisions>

<specifics>
## Specific Ideas

- CLI should work like: `docker run sarcodec compress input.tif output.npz`
- API server mode: `docker run -p 8000:8000 sarcodec --serve`
- FastAPI auto-docs at `/docs` for API exploration
- Include docker-compose.yml for easy local development

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope

</deferred>

---

*Phase: 07-deployment*
*Context gathered: 2026-01-30*
