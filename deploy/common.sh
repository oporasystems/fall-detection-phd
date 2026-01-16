#!/bin/bash
# Common functions for fall detection deployment scripts

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_CONFIG="${SCRIPT_DIR}/.deploy-config"

# Print colored status messages
print_status() {
    echo -e "${BLUE}[$1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Load saved connection details
load_config() {
    if [ -f "$DEPLOY_CONFIG" ]; then
        source "$DEPLOY_CONFIG"
        return 0
    fi
    return 1
}

# Save connection details
save_config() {
    cat > "$DEPLOY_CONFIG" << EOF
PI_HOST="$PI_HOST"
PI_USER="$PI_USER"
PI_PASS="$PI_PASS"
EOF
    chmod 600 "$DEPLOY_CONFIG"
    print_success "Credentials saved to .deploy-config"
}

# Prompt for connection details
get_connection_info() {
    echo ""

    # Try to load existing config
    if load_config; then
        echo "Found saved credentials for ${PI_USER}@${PI_HOST}"
        read -p "Use saved credentials? [Y/n]: " use_saved
        use_saved=${use_saved:-Y}

        if [[ "$use_saved" =~ ^[Yy]$ ]]; then
            echo ""
            return
        fi
        echo ""
    fi

    # Prompt for new credentials
    read -p "Enter Raspberry Pi IP address: " PI_HOST
    read -p "Enter username [pi]: " PI_USER
    PI_USER=${PI_USER:-pi}
    read -sp "Enter password: " PI_PASS
    echo ""
    echo ""

    # Offer to save
    read -p "Save credentials for future deployments? [Y/n]: " save_creds
    save_creds=${save_creds:-Y}

    if [[ "$save_creds" =~ ^[Yy]$ ]]; then
        save_config
    fi
    echo ""
}

# Test SSH connection
test_connection() {
    print_status "1/6" "Testing connection to ${PI_USER}@${PI_HOST}..."
    if sshpass -p "$PI_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${PI_USER}@${PI_HOST}" "echo 'Connected'" > /dev/null 2>&1; then
        print_success "Connection successful"
        return 0
    else
        print_error "Failed to connect to ${PI_USER}@${PI_HOST}"
        return 1
    fi
}

# Run command on Pi via SSH
run_on_pi() {
    sshpass -p "$PI_PASS" ssh -o StrictHostKeyChecking=no "${PI_USER}@${PI_HOST}" "$1"
}

# Copy file to Pi via SCP
copy_to_pi() {
    local src="$1"
    local dest="$2"
    sshpass -p "$PI_PASS" scp -o StrictHostKeyChecking=no "$src" "${PI_USER}@${PI_HOST}:${dest}"
}

# Install dependencies on Pi (only if not already installed)
install_dependencies() {
    print_status "2/6" "Checking dependencies..."

    # Check if i2c-tools is installed
    if ! run_on_pi "dpkg -s i2c-tools > /dev/null 2>&1"; then
        echo "      Installing system packages..."
        run_on_pi "sudo apt-get update -qq"
        run_on_pi "sudo apt-get install -y -qq i2c-tools python3-pip"
    else
        echo "      System packages already installed"
    fi

    # Check if Python packages are installed by testing a key import
    if ! run_on_pi "python3 -c 'import torch; import performer_pytorch' 2>/dev/null"; then
        echo "      Installing Python packages (this may take a while on first run)..."
        echo "      Note: Installing one at a time to avoid out-of-memory on Pi"

        # Install packages one at a time with --no-cache-dir to save memory
        local packages="pandas smbus numpy scikit-learn scipy RPi.GPIO board Adafruit-Blinka adafruit-circuitpython-bmp3xx"
        for pkg in $packages; do
            echo "        - $pkg"
            run_on_pi "pip install --quiet --no-cache-dir $pkg --break-system-packages 2>/dev/null || pip install --quiet --no-cache-dir $pkg" || true
        done

        # Install torch separately (heaviest package)
        echo "        - torch (this one takes longest)"
        run_on_pi "pip install --quiet --no-cache-dir torch --break-system-packages 2>/dev/null || pip install --quiet --no-cache-dir torch" || true

        # Install performer-pytorch last (depends on torch)
        echo "        - performer-pytorch"
        run_on_pi "pip install --quiet --no-cache-dir performer-pytorch --break-system-packages 2>/dev/null || pip install --quiet --no-cache-dir performer-pytorch" || true
    else
        echo "      Python packages already installed"
    fi

    print_success "Dependencies ready"
}

# Upload files to Pi
upload_file() {
    local src="$1"
    local filename=$(basename "$src")
    echo "      - $filename"
    copy_to_pi "$src" "/home/${PI_USER}/${filename}"
}

# Run calibration on Pi
run_calibration() {
    print_status "4/6" "Running sensor calibration..."
    echo ""
    print_warning "Place the device on a flat, level surface"
    print_warning "Keep it completely stationary during calibration"
    echo ""
    read -p "Press ENTER when ready..."
    echo ""
    echo "      Calibrating..."

    # Upload and run calibration script
    copy_to_pi "${PROJECT_ROOT}/iot/pi/calibrate-mpu-6050.py" "/home/${PI_USER}/calibrate-mpu-6050.py"
    run_on_pi "cd /home/${PI_USER} && python3 calibrate-mpu-6050.py"

    print_success "Calibration complete. Offsets saved to /home/${PI_USER}/mpu_offsets.json"
}

# Stop and remove any existing fall-detection services
remove_existing_services() {
    print_status "5/6" "Setting up systemd service..."

    for service in fall-detector adl-collector fall-collector; do
        if run_on_pi "systemctl list-unit-files | grep -q ${service}.service" 2>/dev/null; then
            echo "      Found existing service: ${service}"
            echo "      Stopping and removing..."
            run_on_pi "sudo systemctl stop ${service} 2>/dev/null || true"
            run_on_pi "sudo systemctl disable ${service} 2>/dev/null || true"
            run_on_pi "sudo rm -f /etc/systemd/system/${service}.service"
        fi
    done
    run_on_pi "sudo systemctl daemon-reload"
}

# Create and enable systemd service
create_service() {
    local service_name="$1"
    local script_name="$2"
    local description="$3"

    # Create service file content
    local service_content="[Unit]
Description=${description}
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/${PI_USER}/${script_name}
Restart=always
WorkingDirectory=/home/${PI_USER}
Environment=\"PYTHONUNBUFFERED=1\"

[Install]
WantedBy=multi-user.target"

    # Write service file on Pi
    run_on_pi "echo '${service_content}' | sudo tee /etc/systemd/system/${service_name}.service > /dev/null"
    run_on_pi "sudo systemctl daemon-reload"
    run_on_pi "sudo systemctl enable ${service_name}"

    print_success "Service '${service_name}' created and enabled"
}

# Ask to start service
prompt_start_service() {
    local service_name="$1"

    print_status "6/6" "Deployment complete!"
    echo ""
    read -p "Start the service now? [Y/n]: " start_now
    start_now=${start_now:-Y}

    if [[ "$start_now" =~ ^[Yy]$ ]]; then
        run_on_pi "sudo systemctl start ${service_name}"
        print_success "Service started"
        echo ""
        echo "Check status with:"
        echo "  ssh ${PI_USER}@${PI_HOST} 'systemctl status ${service_name}'"
        echo ""
        echo "View logs with:"
        echo "  ssh ${PI_USER}@${PI_HOST} 'journalctl -u ${service_name} -f'"
    else
        echo ""
        echo "Start manually with:"
        echo "  ssh ${PI_USER}@${PI_HOST} 'sudo systemctl start ${service_name}'"
    fi
}

# Check if sshpass is installed
check_sshpass() {
    if ! command -v sshpass &> /dev/null; then
        print_error "sshpass is not installed"
        echo ""
        echo "Install it with:"
        echo "  macOS:  brew install hudochenkov/sshpass/sshpass"
        echo "  Ubuntu: sudo apt install sshpass"
        echo ""
        exit 1
    fi
}
