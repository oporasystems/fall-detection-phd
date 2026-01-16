#!/bin/bash
# Setup swap on Raspberry Pi
# Run this if deployment fails due to out-of-memory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

print_banner
echo "=== Swap Setup ==="
echo "Creates 1GB swap file on Raspberry Pi"
echo ""

check_sshpass
get_connection_info
test_connection

print_status "2/2" "Setting up swap..."

# Create a small script to run on Pi
SWAP_SCRIPT='#!/bin/bash
echo "Disabling existing swap..."
sudo swapoff -a 2>/dev/null || true

echo "Removing old swapfile..."
sudo rm -f /swapfile

echo "Creating 1GB swapfile..."
sudo dd if=/dev/zero of=/swapfile bs=1M count=1024 status=progress

echo "Setting permissions..."
sudo chmod 600 /swapfile

echo "Formatting swap..."
sudo mkswap /swapfile

echo "Enabling swap..."
sudo swapon /swapfile

echo "Adding to fstab..."
grep -q "/swapfile" /etc/fstab || echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab

echo ""
echo "Done! Current memory:"
free -h
'

# Copy script to Pi and execute
echo "$SWAP_SCRIPT" | sshpass -p "$PI_PASS" ssh -o StrictHostKeyChecking=no "${PI_USER}@${PI_HOST}" "cat > /tmp/setup-swap.sh && chmod +x /tmp/setup-swap.sh && /tmp/setup-swap.sh"

print_success "Swap setup complete!"
echo ""
echo "Now run:"
echo "  ./deploy/deploy-fall-detector.sh"
