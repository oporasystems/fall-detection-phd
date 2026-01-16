#!/bin/bash
# Quick redeploy fall detector - skips dependencies and calibration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

print_banner
echo "=== Fall Detector Redeploy ==="
echo "Quick update: uploads files and restarts service"
echo ""

check_sshpass
get_connection_info
test_connection

# Upload files
print_status "2/4" "Uploading files..."
upload_file "${PROJECT_ROOT}/iot/pi/fall-detector.py"
upload_file "${PROJECT_ROOT}/iot/pi/logging_config.py"
upload_file "${PROJECT_ROOT}/training/variations/performer/performer_model.pt"

# Update service file
print_status "3/4" "Updating service..."
create_service "fall-detector" "fall-detector.py" "Fall Detection Service"

# Restart service
print_status "4/4" "Restarting service..."
run_on_pi "sudo systemctl restart fall-detector"

print_success "Redeploy complete"
echo ""
echo "View logs with:"
echo "  ./deploy/logs.sh -f"
