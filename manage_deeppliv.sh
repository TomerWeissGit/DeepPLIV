#!/bin/bash
# Quick management script for DeepPLIV service

case "$1" in
    start)
        echo "Starting DeepPLIV service..."
        sudo systemctl start deeppliv.service
        sudo systemctl status deeppliv.service --no-pager
        ;;
    stop)
        echo "Stopping DeepPLIV service..."
        sudo systemctl stop deeppliv.service
        sudo systemctl status deeppliv.service --no-pager
        ;;
    restart)
        echo "Restarting DeepPLIV service..."
        sudo systemctl restart deeppliv.service
        sudo systemctl status deeppliv.service --no-pager
        ;;
    status)
        sudo systemctl status deeppliv.service --no-pager -l
        ;;
    logs)
        echo "Showing DeepPLIV logs (Ctrl+C to exit)..."
        sudo journalctl -u deeppliv.service -f
        ;;
    tail)
        echo "Last 50 log entries:"
        sudo journalctl -u deeppliv.service -n 50 --no-pager
        ;;
    enable)
        echo "Enabling DeepPLIV service for auto-start..."
        sudo systemctl enable deeppliv.service
        ;;
    disable)
        echo "Disabling DeepPLIV service auto-start..."
        sudo systemctl disable deeppliv.service
        ;;
    *)
        echo "DeepPLIV Service Management"
        echo "Usage: $0 {start|stop|restart|status|logs|tail|enable|disable}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the DeepPLIV service"
        echo "  stop     - Stop the DeepPLIV service"
        echo "  restart  - Restart the DeepPLIV service"
        echo "  status   - Show current service status"
        echo "  logs     - Follow live logs (Ctrl+C to exit)"
        echo "  tail     - Show last 50 log entries"
        echo "  enable   - Enable auto-start on boot"
        echo "  disable  - Disable auto-start on boot"
        exit 1
        ;;
esac