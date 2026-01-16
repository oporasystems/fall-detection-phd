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

# Print ASCII banner with running then falling animation
print_banner() {
    local ground="~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    # Save cursor position and draw
    echo ""
    tput sc  # Save cursor position

    # Run across the screen
    for pos in 0 4 8 12 16 20; do
        tput rc  # Restore cursor position
        local pad=""
        for ((i=0; i<pos; i++)); do pad+=" "; done
        echo -e "${BLUE}${pad}  O                              ${NC}"
        echo -e "${BLUE}${pad} /|\\                             ${NC}"
        echo -e "${BLUE}${pad} / \\                             ${NC}"
        echo -e "${BLUE}${ground}${NC}"
        sleep 0.1
    done

    # Trip
    tput rc
    echo -e "${YELLOW}                          O/             ${NC}"
    echo -e "${YELLOW}                         /|              ${NC}"
    echo -e "${YELLOW}                         / \\             ${NC}"
    echo -e "${BLUE}${ground}${NC}"
    sleep 0.15

    # Falling
    tput rc
    echo -e "${YELLOW}                           \\O            ${NC}"
    echo -e "${YELLOW}                            |\\           ${NC}"
    echo -e "${YELLOW}                           / \\           ${NC}"
    echo -e "${BLUE}${ground}${NC}"
    sleep 0.15

    # Face down
    tput rc
    echo -e "${RED}                                          ${NC}"
    echo -e "${RED}                          \\o____          ${NC}"
    echo -e "${RED}                           |              ${NC}"
    echo -e "${BLUE}${ground}${NC}"
    sleep 0.4

    echo ""
    echo -e "${GREEN}       FALL DETECTION SYSTEM${NC}"
    echo ""
}

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
        echo "Using saved credentials: ${PI_USER}@${PI_HOST}"
        echo ""
        return
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
    sshpass -p "$PI_PASS" ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=60 -o ServerAliveCountMax=10 "${PI_USER}@${PI_HOST}" "$1"
}

# Copy file to Pi via SCP
copy_to_pi() {
    local src="$1"
    local dest="$2"
    sshpass -p "$PI_PASS" scp -o StrictHostKeyChecking=no "$src" "${PI_USER}@${PI_HOST}:${dest}"
}

# Setup swap space on Pi for installing heavy packages like torch
setup_swap() {
    print_status "2/7" "Checking swap space..."

    # Check current swap in MB
    local swap_total=$(run_on_pi "free -m | grep Swap | awk '{print \$2}'" 2>/dev/null || echo "0")

    if [ "$swap_total" -lt 512 ] 2>/dev/null; then
        echo "      Current swap: ${swap_total}MB - setting up swap..."

        # Run all swap commands in one SSH session to avoid connection issues
        run_on_pi "
            if [ ! -f /swapfile ]; then
                echo 'Creating swap file...'
                sudo fallocate -l 1G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1M count=1024
                sudo chmod 600 /swapfile
                sudo mkswap /swapfile
            fi
            sudo swapon /swapfile 2>/dev/null || true
            grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null
            free -m | grep Swap
        " || print_warning "Swap setup had issues, continuing anyway..."

        print_success "Swap setup complete"
    else
        echo "      Swap sufficient: ${swap_total}MB"
    fi
}

# Install dependencies on Pi (only if not already installed)
install_dependencies() {
    print_status "3/7" "Checking dependencies..."

    # Check if i2c-tools is installed
    if ! run_on_pi "dpkg -s i2c-tools > /dev/null 2>&1"; then
        echo "      Installing system packages..."
        run_on_pi "sudo apt-get update -qq"
        run_on_pi "sudo apt-get install -y -qq i2c-tools python3-pip"
    else
        echo "      System packages already installed"
    fi

    # Check and install Python packages one at a time
    echo "      Checking Python packages..."

    # Order matters: torch before performer-pytorch
    local packages="pandas smbus numpy scikit-learn scipy RPi.GPIO board Adafruit-Blinka adafruit-circuitpython-bmp3xx torch performer-pytorch"

    # Create temp dir on main disk (default /tmp is RAM-based and too small)
    run_on_pi "mkdir -p ~/pip-tmp"

    for pkg in $packages; do
        # Use pip show instead of python import (uses less memory)
        if run_on_pi "pip show $pkg >/dev/null 2>&1"; then
            echo "        ✓ $pkg (installed)"
        else
            echo "        ✗ $pkg (missing) - installing..."
            run_on_pi "TMPDIR=~/pip-tmp pip install --no-cache-dir $pkg --break-system-packages" || true
        fi
    done

    # Clean up temp dir
    run_on_pi "rm -rf ~/pip-tmp"

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
    print_status "5/7" "Checking calibration..."

    # Check if already calibrated
    if run_on_pi "test -f /home/${PI_USER}/mpu_offsets.json" 2>/dev/null; then
        echo "      Calibration file found"
        read -p "      Re-calibrate? [y/N]: " recalibrate
        recalibrate=${recalibrate:-N}
        if [[ ! "$recalibrate" =~ ^[Yy]$ ]]; then
            print_success "Using existing calibration"
            return
        fi
    fi

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
    print_status "6/7" "Setting up systemd service..."

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
User=${PI_USER}
Environment=\"PYTHONUNBUFFERED=1\"
Environment=\"PATH=/home/${PI_USER}/.local/bin:/usr/local/bin:/usr/bin:/bin\"
Environment=\"PYTHONPATH=/home/${PI_USER}/.local/lib/python3.13/site-packages\"

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

    print_status "7/7" "Deployment complete!"
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
