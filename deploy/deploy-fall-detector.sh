#!/bin/bash
# Deploy fall detector to Raspberry Pi
# This is the production deployment for real-time fall detection

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

echo ""
echo "=== Fall Detector Deployment ==="
echo "Deploys real-time fall detection for production use"
echo ""

# Check prerequisites
check_sshpass

# Get connection info
get_connection_info

# Test connection
test_connection

# Install dependencies
install_dependencies

# Upload files
print_status "3/6" "Uploading files..."
upload_file "${PROJECT_ROOT}/iot/pi/fall-detector.py"
upload_file "${PROJECT_ROOT}/training/variations/performer/performer_model.pt"

# Run calibration
run_calibration

# Setup service
remove_existing_services
create_service "fall-detector" "fall-detector.py" "Fall Detection Service"

# Done
prompt_start_service "fall-detector"
