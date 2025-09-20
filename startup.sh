#!/bin/bash
# EC2 User Data Script for DeepPLIV Auto-Start
# This script runs automatically when a new spot instance starts

set -e  # Exit on any error

# Logging setup
LOG_FILE="/var/log/deeppliv-startup.log"
exec > >(tee -a $LOG_FILE)
exec 2>&1

echo "=================================================================="
echo "DeepPLIV Startup Script - $(date)"
echo "=================================================================="

# Navigate to project directory
PROJECT_DIR="/home/ubuntu/DeepPLIV"
echo "Changing to project directory: $PROJECT_DIR"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory $PROJECT_DIR not found!"
    exit 1
fi

cd $PROJECT_DIR

# Check if we're running as root (User Data runs as root)
echo "Running as user: $(whoami)"

# Update system packages (minimal)
echo "Updating package lists..."
apt-get update -q

# Install Python3 and pip if not available
echo "Ensuring Python3 and pip are available..."
apt-get install -y python3 python3-pip python3-venv

# Create virtual environment if it doesn't exist
VENV_DIR="/home/ubuntu/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv $VENV_DIR
    chown -R ubuntu:ubuntu $VENV_DIR
fi

# Activate virtual environment and install requirements
echo "Installing Python requirements..."
sudo -u ubuntu bash << 'EOF'
source /home/ubuntu/venv/bin/activate
cd /home/ubuntu/DeepPLIV

# Upgrade pip first
pip install --upgrade pip

# Install requirements
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

echo "Python environment setup complete!"
EOF

# Create systemd service for auto-restart
echo "Setting up systemd service..."

# Create service file
cat > /etc/systemd/system/deeppliv.service << 'EOF'
[Unit]
Description=DeepPLIV Genetic IV Simulation
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/DeepPLIV
Environment=PATH=/home/ubuntu/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/ubuntu/venv/bin/python aws_genetic_iv_parallel.py execute
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=deeppliv

# Resource limits
LimitNOFILE=65536
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

# Set proper permissions
chown ubuntu:ubuntu /home/ubuntu/DeepPLIV -R

# Reload systemd and enable service
echo "Enabling and starting DeepPLIV service..."
systemctl daemon-reload
systemctl enable deeppliv.service

# Wait a moment for everything to settle
sleep 5

# Start the service
echo "Starting DeepPLIV execution..."
systemctl start deeppliv.service

# Check service status
echo "Service status:"
systemctl status deeppliv.service --no-pager -l

echo "=================================================================="
echo "DeepPLIV startup completed successfully!"
echo "=================================================================="
echo "Monitor logs with: sudo journalctl -u deeppliv.service -f"
echo "Check status with: sudo systemctl status deeppliv.service"
echo "=================================================================="