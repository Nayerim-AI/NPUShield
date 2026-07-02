#!/usr/bin/env bash
# Install NPUShield as a systemd service.
# Usage: bash deploy/install-service.sh [--model /path/to/model.rkllm] [--api-key YOUR_KEY]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/npushield.service"
INSTALL_PATH="/etc/systemd/system/npushield.service"
ENV_DIR="/etc/npushield"
ENV_FILE="$ENV_DIR/env"

MODEL_PATH=""
API_KEY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL_PATH="$2"; shift 2 ;;
        --api-key) API_KEY="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "==> Installing NPUShield systemd service"

sudo cp "$SERVICE_FILE" "$INSTALL_PATH"
sudo mkdir -p "$ENV_DIR"
sudo chmod 700 "$ENV_DIR"

{
    echo "NPUSHIELD_HOST=0.0.0.0"
    echo "NPUSHIELD_PORT=18999"
    [[ -n "$MODEL_PATH" ]] && echo "NPUSHIELD_RKLLM_MODEL=$MODEL_PATH"
    [[ -n "$API_KEY" ]]    && echo "NPUSHIELD_API_KEY=$API_KEY"
} | sudo tee "$ENV_FILE" > /dev/null
sudo chmod 600 "$ENV_FILE"

sudo sed -i \
    's|# EnvironmentFile=/etc/npushield/env|EnvironmentFile=/etc/npushield/env|' \
    "$INSTALL_PATH"
sudo sed -i \
    '/^Environment=NPUSHIELD_HOST=/d
     /^Environment=NPUSHIELD_PORT=/d
     /^# Environment=NPUSHIELD_API_KEY/d
     /^# Environment=NPUSHIELD_RKLLM_MODEL/d' \
    "$INSTALL_PATH"

sudo systemctl daemon-reload
sudo systemctl enable npushield

echo ""
echo "==> Done. Commands:"
echo "    sudo systemctl start npushield"
echo "    sudo journalctl -u npushield -f"
echo "    sudo nano $ENV_FILE      # set model path / API key"
