#!/bin/bash
# One-time setup script to install DeepPLIV as a systemd service
# Run this once on your instance before creating the AMI or using User Data

set -e

echo "=================================================================="
echo "DeepPLIV Service Installation Script"
echo "=================================================================="

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    echo "This script should be run as ubuntu user, not root."
    echo "Usage: ./install_service.sh"
    exit 1
fi

PROJECT_DIR="/home/ubuntu/DeepPLIV"
VENV_DIR="/home/ubuntu/venv"

# Verify we're in the right directory
if [ ! -f "$PROJECT_DIR/deeppliv.service" ]; then
    echo "ERROR: deeppliv.service file not found in $PROJECT_DIR"
    echo "Please make sure you're running this from the DeepPLIV directory"
    exit 1
fi

echo "Installing DeepPLIV systemd service..."

# Install required packages
echo "Installing required system packages..."
sudo apt update -q
sudo apt install -y python3-pip

# Install requirements globally (override Ubuntu 24.04 protection)
echo "Installing Python requirements..."
pip3 install -r requirements.txt

# Copy service file to systemd directory (requires sudo)
echo "Installing systemd service file..."
sudo cp deeppliv.service /etc/systemd/system/

# Set proper permissions
sudo chown root:root /etc/systemd/system/deeppliv.service
sudo chmod 644 /etc/systemd/system/deeppliv.service

# Reload systemd daemon
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable service (but don't start it yet)
echo "Enabling DeepPLIV service..."
sudo systemctl enable deeppliv.service

echo "=================================================================="
echo "Service installation completed successfully!"
echo "=================================================================="
echo ""
echo "Available commands:"
echo "  Start service:    sudo systemctl start deeppliv.service"
echo "  Stop service:     sudo systemctl stop deeppliv.service"
echo "  Check status:     sudo systemctl status deeppliv.service"
echo "  View logs:        sudo journalctl -u deeppliv.service -f"
echo "  Restart service:  sudo systemctl restart deeppliv.service"
echo ""
echo "The service is now enabled and will start automatically on boot."
echo "=================================================================="