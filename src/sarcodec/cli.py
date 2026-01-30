"""
SAR Codec CLI entry point.

This module provides the entry point for the `sarcodec` command
installed via pip.
"""

import sys
from pathlib import Path

# Add project root to path if running from source
_project_root = Path(__file__).parent.parent.parent
if _project_root.exists():
    sys.path.insert(0, str(_project_root))


def main():
    """Main entry point for sarcodec CLI."""
    # Import and run the actual CLI
    from scripts.sarcodec import main as cli_main
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
