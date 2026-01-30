#!/bin/bash
set -e

# SAR Codec Docker Entrypoint
# Handles both CLI and API server modes

# Function to show help
show_help() {
    echo "SAR Codec - SAR Image Compression Tool"
    echo ""
    echo "Usage:"
    echo "  docker run sarcodec [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  compress INPUT OUTPUT   Compress GeoTIFF to NPZ"
    echo "  decompress INPUT OUTPUT Decompress NPZ to GeoTIFF"
    echo "  --serve                 Start REST API server"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Compress a file (mount data directory)"
    echo "  docker run -v \$(pwd)/data:/data sarcodec compress /data/input.tif /data/output.npz"
    echo ""
    echo "  # Start API server"
    echo "  docker run -p 8000:8000 --gpus all sarcodec --serve"
    echo ""
    echo "  # API server with custom models"
    echo "  docker run -p 8000:8000 -v \$(pwd)/models:/app/models --gpus all sarcodec --serve"
    echo ""
    echo "Environment variables:"
    echo "  SARCODEC_CHECKPOINT_DIR  Model directory (default: /app/models)"
    echo "  SARCODEC_HOST            API host (default: 0.0.0.0)"
    echo "  SARCODEC_PORT            API port (default: 8000)"
}

# Check first argument
case "${1:-}" in
    --serve)
        # Start API server
        echo "Starting SAR Codec API server..."
        echo "  Host: ${SARCODEC_HOST:-0.0.0.0}"
        echo "  Port: ${SARCODEC_PORT:-8000}"
        echo "  Models: ${SARCODEC_CHECKPOINT_DIR:-/app/models}"
        exec python -m uvicorn src.api.app:app \
            --host "${SARCODEC_HOST:-0.0.0.0}" \
            --port "${SARCODEC_PORT:-8000}"
        ;;

    compress|decompress)
        # Run CLI command
        exec python scripts/sarcodec.py "$@"
        ;;

    --help|-h|"")
        show_help
        exit 0
        ;;

    *)
        # Unknown command - try passing to sarcodec CLI
        exec python scripts/sarcodec.py "$@"
        ;;
esac
