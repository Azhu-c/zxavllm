#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_HOME=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec python3 "$EXPERIMENT_HOME/experiment.py" "$@"
