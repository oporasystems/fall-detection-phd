#!/bin/bash
# Quick redeploy fall collector - skips dependencies and calibration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

echo ""
echo "=== Fall Collector Redeploy ==="
echo "Quick update: uploads files and restarts service"
echo ""

check_sshpass
get_connection_info
test_connection

# Upload files
print_status "2/3" "Uploading files..."
upload_file "${PROJECT_ROOT}/iot/pi/data-collector-falls.py"
upload_file "${PROJECT_ROOT}/iot/pi/logging_config.py"

# Upload heatmap if it exists
HEATMAP_FILE="${PROJECT_ROOT}/dataset/visualisation/heatmap_array.json"
if [ -f "$HEATMAP_FILE" ]; then
    upload_file "$HEATMAP_FILE"
fi

# Restart service
print_status "3/3" "Restarting service..."
run_on_pi "sudo systemctl restart fall-collector"

print_success "Redeploy complete"
echo ""
echo "View logs with:"
echo "  ./deploy/logs.sh -f"
