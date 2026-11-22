#!/usr/bin/env bash
set -euo pipefail

uv sync
uv run plot-lm-metrics
uv run plot-paraphrase-consistency
