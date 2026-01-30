# API Reference

REST API documentation for SAR Codec compression service.

## Overview

The SAR Codec API provides HTTP endpoints for compressing and decompressing SAR images using trained autoencoder models.

- **Base URL:** `http://localhost:8000`
- **Authentication:** None required
- **Content-Type:** `multipart/form-data` for uploads

## Endpoints

### GET /health

Health check endpoint returning API status and loaded models.

**Request:**

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": ["4x", "8x", "16x"],
  "device": "cuda",
  "cuda_available": true
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"healthy"` if models loaded, `"degraded"` otherwise |
| `version` | string | API version |
| `models_loaded` | array | List of available model variants |
| `device` | string | Inference device (`"cuda"` or `"cpu"`) |
| `cuda_available` | boolean | Whether CUDA is available |

---

### POST /encode

Compress a GeoTIFF image to NPZ latent representation.

**Request:**

```bash
curl -X POST "http://localhost:8000/encode?model=8x" \
    -F "file=@input.tif" \
    -o compressed.npz
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | query | No | `"8x"` | Model variant: `"4x"`, `"8x"`, or `"16x"` |
| `file` | form-data | Yes | - | GeoTIFF file to compress |

**Response:**

- **Content-Type:** `application/octet-stream`
- **Body:** NPZ file containing latent representation and metadata

**Response Headers:**

| Header | Description |
|--------|-------------|
| `Content-Disposition` | Suggested filename for download |
| `X-Processing-Time-Ms` | Processing time in milliseconds |

**Example with Python:**

```python
import requests

with open("input.tif", "rb") as f:
    response = requests.post(
        "http://localhost:8000/encode?model=8x",
        files={"file": ("input.tif", f, "image/tiff")}
    )

with open("compressed.npz", "wb") as f:
    f.write(response.content)

print(f"Processing time: {response.headers['X-Processing-Time-Ms']} ms")
```

---

### POST /decode

Decompress an NPZ file back to GeoTIFF format.

**Request:**

```bash
curl -X POST "http://localhost:8000/decode?model=8x" \
    -F "file=@compressed.npz" \
    -o reconstructed.tif
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | query | No | `"8x"` | Model variant (must match encode model) |
| `file` | form-data | Yes | - | NPZ file from `/encode` endpoint |

**Response:**

- **Content-Type:** `image/tiff`
- **Body:** Reconstructed GeoTIFF image

**Response Headers:**

| Header | Description |
|--------|-------------|
| `Content-Disposition` | Suggested filename for download |
| `X-Processing-Time-Ms` | Processing time in milliseconds |

**Example with Python:**

```python
import requests

with open("compressed.npz", "rb") as f:
    response = requests.post(
        "http://localhost:8000/decode?model=8x",
        files={"file": ("compressed.npz", f, "application/octet-stream")}
    )

with open("reconstructed.tif", "wb") as f:
    f.write(response.content)
```

---

### POST /compress

Alias for `/encode`. Compress GeoTIFF to NPZ.

```bash
curl -X POST "http://localhost:8000/compress?model=8x" \
    -F "file=@input.tif" \
    -o compressed.npz
```

---

### POST /decompress

Alias for `/decode`. Decompress NPZ to GeoTIFF.

```bash
curl -X POST "http://localhost:8000/decompress?model=8x" \
    -F "file=@compressed.npz" \
    -o reconstructed.tif
```

---

## Error Responses

All errors return JSON with a `detail` field.

### 400 Bad Request

Invalid file format.

```json
{
  "detail": "File must be GeoTIFF (.tif or .tiff)"
}
```

### 404 Not Found

Requested model not available.

```json
{
  "detail": "Model '4x' not loaded. Available models: 8x, 16x"
}
```

### 500 Internal Server Error

Processing error.

```json
{
  "detail": "Compression failed: CUDA out of memory"
}
```

### Error Handling Example

```python
import requests

response = requests.post(
    "http://localhost:8000/encode?model=32x",  # Invalid model
    files={"file": open("input.tif", "rb")}
)

if response.status_code != 200:
    error = response.json()
    print(f"Error {response.status_code}: {error['detail']}")
else:
    # Save compressed file
    with open("output.npz", "wb") as f:
        f.write(response.content)
```

---

## Model Variants

| Model | Compression Ratio | PSNR | Latent Size | Use Case |
|-------|-------------------|------|-------------|----------|
| `4x` | 4:1 | 24.95 dB | 64 channels | Quality-critical |
| `8x` | 8:1 | 23.13 dB | 32 channels | Balanced (default) |
| `16x` | 16:1 | 20.52 dB | 16 channels | Maximum compression |

---

## Rate Limiting

No rate limiting is applied by default. For production deployments, consider adding a reverse proxy (nginx, traefik) with rate limiting.

---

## Interactive Documentation

The API provides automatic interactive documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

These interfaces allow you to explore endpoints and make test requests directly from the browser.
