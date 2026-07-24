#!/bin/bash
set -e

echo "Starting DocVision AI Container Entrypoint..."

# Create runtime directories if missing
mkdir -p /app/datasets /app/logs /app/outputs /app/models

# Set permission safety
chmod -R 777 /app/outputs /app/logs /app/datasets /app/models 2>/dev/null || true

# Execute the main container payload command
exec "$@"
