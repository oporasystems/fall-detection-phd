#!/bin/bash
# Deploy fall data collector to Raspberry Pi
# Use this to collect fall samples for training (with heatmap timing)

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

print_banner
echo "=== Fall Data Collector Deployment ==="
echo "Deploys data collection for fall events with timing heatmap"
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
upload_file "${PROJECT_ROOT}/iot/pi/data-collector-falls.py"
upload_file "${PROJECT_ROOT}/iot/pi/logging_config.py"

# Upload heatmap if it exists
HEATMAP_FILE="${PROJECT_ROOT}/dataset/visualisation/heatmap_array.json"
if [ -f "$HEATMAP_FILE" ]; then
    upload_file "$HEATMAP_FILE"
    print_success "Heatmap file uploaded"
else
    print_warning "No existing heatmap found, device will start with empty heatmap"
fi

# Run calibration
run_calibration

# Setup service
remove_existing_services
create_service "fall-collector" "data-collector-falls.py" "Fall Data Collection Service"

# Done
prompt_start_service "fall-collector"
