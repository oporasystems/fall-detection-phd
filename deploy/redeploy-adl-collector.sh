#!/bin/bash
# Quick redeploy ADL collector - skips dependencies and calibration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

echo ""
echo "=== ADL Collector Redeploy ==="
echo "Quick update: uploads files and restarts service"
echo ""

check_sshpass
get_connection_info
test_connection

# Upload files
print_status "2/3" "Uploading files..."
upload_file "${PROJECT_ROOT}/iot/pi/data-collector-adl.py"
upload_file "${PROJECT_ROOT}/iot/pi/logging_config.py"

# Restart service
print_status "3/3" "Restarting service..."
run_on_pi "sudo systemctl restart adl-collector"

print_success "Redeploy complete"
echo ""
echo "View logs with:"
echo "  ./deploy/logs.sh -f"
