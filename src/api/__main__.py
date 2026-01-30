"""
Entry point for running the API server.

Usage:
    python -m src.api [--host HOST] [--port PORT]
"""

import uvicorn


def main():
    """Run the FastAPI server."""
    import argparse

    parser = argparse.ArgumentParser(description="SAR Codec API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
