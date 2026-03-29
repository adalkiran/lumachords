#!/usr/bin/env bash
set -euo pipefail

uv run pyinstaller -y --clean --noconfirm \
  lumachords.spec
