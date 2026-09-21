# DeepPLIV Spot Instance Auto-Restart Setup

This setup enables your DeepPLIV simulation to automatically restart and resume from checkpoints when spot instances are interrupted.

## Files Created

1. **`startup.sh`** - EC2 User Data script for automatic setup
2. **`deeppliv.service`** - Systemd service definition
3. **`install_service.sh`** - One-time service installation script
4. **`manage_deeppliv.sh`** - Service management utility

## Setup Options

### Option A: Use EC2 User Data (Recommended for New Instances)

1. **Copy the startup.sh content** to EC2 User Data when launching your spot instance
2. **Launch your spot instance** - everything will set up automatically
3. **Monitor progress** with:
   ```bash
   tail -f /var/log/deeppliv-startup.log
   sudo journalctl -u deeppliv.service -f
   ```

### Option B: Manual Setup (For Existing Instances)

1. **Make scripts executable:**
   ```bash
   chmod +x install_service.sh manage_deeppliv.sh
   ```

2. **Install the service:**
   ```bash
   ./install_service.sh
   ```

3. **Start the service:**
   ```bash
   ./manage_deeppliv.sh start
   ```

## Service Management

Use the management script for easy control:

```bash
# Start/stop/restart
./manage_deeppliv.sh start
./manage_deeppliv.sh stop
./manage_deeppliv.sh restart

# Monitor
./manage_deeppliv.sh status
./manage_deeppliv.sh logs      # Follow live logs
./manage_deeppliv.sh tail      # Last 50 entries

# Auto-start control
./manage_deeppliv.sh enable    # Enable auto-start on boot
./manage_deeppliv.sh disable   # Disable auto-start
```

## EC2 User Data Setup

When launching your spot instance, use this as User Data:

```bash
#!/bin/bash
# Download and run the startup script
cd /home/ubuntu/DeepPLIV
bash startup.sh
```

Or copy the entire contents of `startup.sh` into the User Data field.

## How It Works

### Automatic Restart Flow:
1. **Spot instance starts** → User Data runs
2. **Install packages** → Create virtual environment + pip install
3. **Setup service** → Install systemd service
4. **Start execution** → Run `python aws_genetic_iv_parallel.py execute`
5. **Resume from checkpoint** → Script automatically loads S3 checkpoint
6. **Auto-restart on failure** → Service restarts if script crashes

### Checkpoint Recovery:
- Every completed task saves progress to S3
- New instances automatically resume from last checkpoint
- Maximum 1 task lost per interruption

## Monitoring

### View Current Status:
```bash
sudo systemctl status deeppliv.service
```

### Follow Real-time Logs:
```bash
sudo journalctl -u deeppliv.service -f
```

### Check Startup Logs:
```bash
tail -f /var/log/deeppliv-startup.log
```

## Troubleshooting

### Service Won't Start:
```bash
# Check service status
sudo systemctl status deeppliv.service -l

# Check startup logs
cat /var/log/deeppliv-startup.log

# Check if requirements are installed
source /home/ubuntu/venv/bin/activate
pip list
```

### Missing Dependencies:
```bash
# Reinstall requirements
cd /home/ubuntu/DeepPLIV
source /home/ubuntu/venv/bin/activate
pip install -r requirements.txt
```

### Restart Everything:
```bash
sudo systemctl restart deeppliv.service
```

## Spot Instance Best Practices

1. **Use Auto Scaling Group** with spot instances for even better reliability
2. **Monitor CloudWatch logs** for early detection of issues
3. **Set up SNS notifications** for instance termination alerts
4. **Use multiple AZs** to increase spot availability

## Security Notes

- Service runs as `ubuntu` user (not root)
- Private tmp directory for isolation
- Limited file descriptors and processes
- No privilege escalation allowed

## Expected Timeline

- **Instance boot → Service start**: 3-5 minutes
- **Package installation**: 2-3 minutes
- **Script start → Checkpoint load**: 30-60 seconds
- **Total recovery time**: 5-8 minutes per spot interruption

Your 3-week simulation will now run autonomously with automatic recovery!