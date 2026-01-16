#!/bin/bash
# View logs from Raspberry Pi fall detection system

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Default values
LINES=100
FOLLOW=false
DATE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--follow)
            FOLLOW=true
            shift
            ;;
        -n|--lines)
            LINES="$2"
            shift 2
            ;;
        -d|--date)
            DATE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -f, --follow     Follow log output in real-time"
            echo "  -n, --lines N    Show last N lines (default: 100)"
            echo "  -d, --date DATE  Show logs for specific date (format: YYYY-MM-DD)"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Show last 100 lines"
            echo "  $0 -f                 # Follow logs in real-time"
            echo "  $0 -n 500             # Show last 500 lines"
            echo "  $0 -d 2024-01-15      # Show logs from specific date"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

print_banner
echo "=== Fall Detection Logs ==="
echo ""

# Check prerequisites
check_sshpass

# Get connection info
get_connection_info

# Test connection
print_status "1/2" "Connecting to ${PI_USER}@${PI_HOST}..."
if ! sshpass -p "$PI_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${PI_USER}@${PI_HOST}" "echo 'Connected'" > /dev/null 2>&1; then
    print_error "Failed to connect"
    exit 1
fi
print_success "Connected"

# Determine log file path
LOG_DIR="/home/${PI_USER}/logs"
LOG_FILE="${LOG_DIR}/fall-detection.log"

if [ -n "$DATE" ]; then
    # Show logs for specific date
    LOG_FILE="${LOG_DIR}/fall-detection.log.${DATE}"
    print_status "2/2" "Fetching logs for ${DATE}..."
    echo ""
    run_on_pi "cat ${LOG_FILE} 2>/dev/null || echo 'No logs found for ${DATE}'"
elif [ "$FOLLOW" = true ]; then
    # Follow logs in real-time
    print_status "2/2" "Following logs (Ctrl+C to stop)..."
    echo ""
    echo "-------------------------------------------"
    sshpass -p "$PI_PASS" ssh -o StrictHostKeyChecking=no "${PI_USER}@${PI_HOST}" "tail -f ${LOG_FILE}"
else
    # Show last N lines
    print_status "2/2" "Fetching last ${LINES} lines..."
    echo ""
    echo "-------------------------------------------"
    run_on_pi "tail -n ${LINES} ${LOG_FILE} 2>/dev/null || echo 'No logs found. Log file: ${LOG_FILE}'"
fi
