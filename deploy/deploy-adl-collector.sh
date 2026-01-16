#!/bin/bash
# Deploy ADL (Activities of Daily Living) data collector to Raspberry Pi
# Use this to collect normal activity samples for training

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

print_banner
echo "=== ADL Data Collector Deployment ==="
echo "Deploys data collection for normal daily activities"
echo ""

# Check prerequisites
check_sshpass

# Get connection info
get_connection_info

# Test connection
test_connection

# Setup swap for heavy packages
setup_swap

# Install dependencies
install_dependencies

# Upload files
print_status "4/7" "Uploading files..."
upload_file "${PROJECT_ROOT}/iot/pi/data-collector-adl.py"
upload_file "${PROJECT_ROOT}/iot/pi/logging_config.py"

# Run calibration
run_calibration

# Setup service
remove_existing_services
create_service "adl-collector" "data-collector-adl.py" "ADL Data Collection Service"

# Done
prompt_start_service "adl-collector"
