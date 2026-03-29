#!/usr/bin/env bash
set -euo pipefail

uv run pyi-makespec \
  --windowed \
  --name LumaChords \
  --paths . \
  lumachords/__main__.py
