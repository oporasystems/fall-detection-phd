# Fall Detection System

A transformer-based fall detection system combining IoT sensors (Raspberry Pi + MPU6050 accelerometer) with deep learning for real-time fall classification. This project was developed as part of PhD research comparing multiple transformer architectures for time-series classification.

## Overview

The system collects accelerometer data at 100Hz, processes it through trained neural network models, and triggers alerts via a GPIO-connected buzzer when a fall is detected. After extensive comparison of 10 different transformer variants, the **Performer** architecture was selected for production deployment due to its optimal balance of accuracy and inference speed on embedded hardware.

### Key Features

- Real-time fall detection on Raspberry Pi Zero 2W
- 100Hz accelerometer sampling with Butterworth filtering
- Binary classification (Fall vs Activities of Daily Living)
- Sub-5MB model size for embedded deployment
- Audible alerts via piezo buzzer

## Hardware Requirements

| Component | Description |
|-----------|-------------|
| Raspberry Pi Zero 2W | Main processing unit |
| MPU6050 | 6-axis accelerometer/gyroscope (I2C address 0x68) |
| BMP388 | Barometric pressure sensor (for altitude) |
| Piezo Buzzer | Connected to GPIO 23 |
| Push Buttons | GPIO 24 (tracking on/off), GPIO 25 (alert on/off) |

### Wiring Diagram

```
MPU6050:
  VCC -> 3.3V
  GND -> GND
  SDA -> GPIO 2 (SDA)
  SCL -> GPIO 3 (SCL)

BMP388:
  VCC -> 3.3V
  GND -> GND
  SDA -> GPIO 2 (SDA)
  SCL -> GPIO 3 (SCL)

Buzzer:
  + -> GPIO 23
  - -> GND

Buttons:
  Button 1 -> GPIO 24 (with pull-up)
  Button 2 -> GPIO 25 (with pull-up)
```

## Quick Start (Raspberry Pi)

### 1. Install Dependencies

```bash
sudo apt install -y i2c-tools python3-pip
pip install pandas smbus numpy scikit-learn torch scipy RPi.GPIO \
    board Adafruit-Blinka adafruit-circuitpython-bmp3xx performer-pytorch \
    --break-system-packages
```

### 2. Enable I2C

```bash
sudo raspi-config
# Navigate to: Interface Options -> I2C -> Enable
```

### 3. Verify Sensors

```bash
i2cdetect -y 1
# Should show devices at 0x68 (MPU6050) and 0x77 (BMP388)
```

### 4. Deploy Model and Config Files

Copy to your Raspberry Pi:
- `performer_model.pt` -> `/home/ivanursul/performer_model.pt`
- `mpu_offsets.json` -> `/home/ivanursul/mpu_offsets.json`

### 5. Run Fall Detector

```bash
python3 iot/pi/fall-detector.py
```

### 6. Setup as Systemd Service (Optional)

Create service file:
```bash
sudo nano /etc/systemd/system/fall-detector.service
```

Contents:
```ini
[Unit]
Description=Fall Detection Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/ivanursul/fall-detector.py
Restart=always
WorkingDirectory=/home/ivanursul
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable fall-detector.service
sudo systemctl start fall-detector.service
```

Check status:
```bash
sudo systemctl status fall-detector.service
journalctl -u fall-detector.service --no-pager --lines=50
```

## Project Structure

```
fall-detection-phd/
├── config.yaml                 # Base configuration (committed)
├── config.local.yaml.example   # Template for local overrides
├── deploy/                         # Deployment scripts
│   ├── common.sh                   # Shared deployment functions
│   ├── deploy-fall-detector.sh     # Deploy production fall detection
│   ├── deploy-adl-collector.sh     # Deploy ADL data collector
│   ├── deploy-fall-collector.sh    # Deploy fall data collector
│   └── logs.sh                     # View logs from Pi remotely
├── iot/
│   ├── pi/
│   │   ├── fall-detector.py        # Production fall detection (uses Performer)
│   │   ├── data-collector-adl.py   # Collect ADL (daily activities) data
│   │   ├── data-collector-falls.py # Collect fall data with timing heatmap
│   │   ├── calibrate-mpu-6050.py   # Sensor calibration utility
│   │   ├── logging_config.py       # Shared logging configuration
│   │   └── debug/                  # Debugging utilities
│   └── arduino/
│       └── converter/              # TFLite model conversion
├── training/
│   ├── models/                     # Model architecture definitions
│   │   ├── transformer.py          # Base transformer
│   │   ├── performer.py            # Performer (PRODUCTION MODEL)
│   │   ├── informer.py             # Sparse attention
│   │   ├── linformer.py            # Linear complexity
│   │   ├── multi_scale_transformer.py
│   │   ├── temporal_convolutional_transformer.py
│   │   ├── lstm_transformer.py
│   │   ├── t2v_bert.py             # Time2Vec + BERT
│   │   └── transformer_laurel.py   # LAuReL residual enhancement
│   ├── variations/                 # Training scripts per model
│   │   ├── performer/              # Performer training & tuning
│   │   ├── transformer/
│   │   ├── informer/
│   │   └── ...                     # Other variants
│   ├── experiments/                # Model comparison experiments
│   └── utils/
│       ├── constants.py            # Configuration loader
│       ├── dataset_utils.py        # Dataset loading
│       ├── train_utils.py          # Training utilities
│       └── logging_utils.py        # Logging setup
├── dataset/
│   ├── review/                     # Dataset review utilities
│   ├── visualisation/              # Visualization scripts
│   └── video/                      # Video-based dataset tools
├── 3d-models/                      # OpenSCAD enclosure designs
├── pcb/                            # Circuit board designs
└── docs/
    └── plans/                      # Design documents
```

## Model Architectures

This project implements and compares 10 transformer variants for fall detection:

| Model | Description | Key Feature |
|-------|-------------|-------------|
| **Performer** | Kernel-based efficient attention | **Selected for production** - best accuracy/speed tradeoff |
| Transformer | Standard multi-head attention | Baseline implementation |
| Informer | ProbSparse self-attention | Reduced complexity for long sequences |
| Linformer | Linear complexity attention | O(n) instead of O(n²) |
| Multi-Scale | Multi-resolution processing | Captures patterns at different time scales |
| TCN-Transformer | Temporal convolution + transformer | Hybrid architecture |
| LSTM-Transformer | LSTM + transformer | Combines recurrent and attention |
| T2V-BERT | Time2Vec + BERT | Learned time representations |
| LAuReL | Residual enhancement | Latest experimental addition |
| Reformer | Local sensitive hashing | Memory-efficient attention |

### Why Performer?

After hyperparameter tuning with Optuna and comparison experiments, Performer was selected because:
1. Achieves comparable accuracy to standard transformer
2. Significantly faster inference on Raspberry Pi
3. Model size under 5MB constraint for embedded deployment
4. Stable training with good generalization

## Training Pipeline

### Dataset Format

The system expects CSV files with accelerometer data:
- **Columns**: `AccX_filtered_i`, `AccY_filtered_i`, `Corrected_AccZ_i`, `Acc_magnitude_i`
- **Sequence length**: 800 timesteps (8 seconds at 100Hz)
- **Labels**: Binary (0 = Non-Fall/ADL, 1 = Fall)

### Configuration

1. Copy the example config:
   ```bash
   cp config.local.yaml.example config.local.yaml
   ```

2. Edit `config.local.yaml` with your dataset paths:
   ```yaml
   dataset:
     fall_folder: "/path/to/your/Fall/data"
     non_fall_folder: "/path/to/your/ADL/data"
   ```

### Training a Model

```bash
# Train Performer model
python3 training/variations/performer/performer-model.py

# Run hyperparameter tuning
python3 training/variations/performer/performer-hyperparameter-tuning.py
```

### Model Comparison

The `training/experiments/transformer-variation-comparison/` directory contains scripts to evaluate all models on the same dataset, measuring:
- Accuracy, Precision, Recall, F1-score
- Inference latency
- Memory usage
- Model size

## Data Collection

### Activities of Daily Living (ADL)

```bash
python3 iot/pi/data-collector-adl.py
```

Collects 8-second windows of normal daily activities. Press the button (GPIO 24) to start/stop recording.

### Fall Data

```bash
python3 iot/pi/data-collector-falls.py
```

Collects fall event data with an intelligent timing system:
- Maintains a heatmap of when falls occur within the 8-second window
- Beeps to signal when to perform the fall
- Ensures balanced dataset distribution across all 100ms intervals

This approach prevents bias in fall timing and improves model generalization.

## Deployment

Automated deployment scripts make it easy to set up Raspberry Pi devices for your team.

### Prerequisites

Install `sshpass` on your local machine for automated SSH:

```bash
# macOS
brew install hudochenkov/sshpass/sshpass

# Ubuntu/Debian
sudo apt install sshpass
```

### Deploy Fall Detector (Production)

```bash
./deploy/deploy-fall-detector.sh
```

Deploys real-time fall detection:
- Uploads `fall-detector.py` and trained Performer model
- Runs sensor calibration
- Sets up auto-start systemd service

### Deploy ADL Data Collector

```bash
./deploy/deploy-adl-collector.sh
```

Deploys data collection for normal daily activities:
- Uploads `data-collector-adl.py`
- Runs sensor calibration
- Sets up systemd service

### Deploy Fall Data Collector

```bash
./deploy/deploy-fall-collector.sh
```

Deploys data collection for fall events:
- Uploads `data-collector-falls.py` and timing heatmap
- Runs sensor calibration
- Sets up systemd service

### Switching Modes

Only one mode can be active at a time. Deploying a new mode automatically stops and replaces the previous one:

1. Deploy `fall-collector` → collect fall data
2. Deploy `adl-collector` → collect daily activity data
3. Retrain model with new data
4. Deploy `fall-detector` → back to production

### Calibration

During deployment, you'll be prompted to position the device:

1. Place the device on a **flat, level surface**
2. Keep it **completely stationary**
3. Press ENTER to start calibration

The calibration measures accelerometer bias and saves offsets to `mpu_offsets.json`.

### Viewing Logs

Logs are stored in `~/logs/fall-detection.log` with daily rotation (30 days retention).

```bash
# View logs interactively
./deploy/logs.sh

# Follow logs in real-time
./deploy/logs.sh -f

# Show last 500 lines
./deploy/logs.sh -n 500

# Show logs from specific date
./deploy/logs.sh -d 2024-01-15
```

### Managing Services

```bash
# Check status
ssh pi@<IP> 'systemctl status fall-detector'

# Restart
ssh pi@<IP> 'sudo systemctl restart fall-detector'

# Stop
ssh pi@<IP> 'sudo systemctl stop fall-detector'
```

## Configuration Reference

### config.yaml

| Section | Key | Description |
|---------|-----|-------------|
| `dataset.fall_folder` | Path to fall data CSVs |
| `dataset.non_fall_folder` | Path to ADL data CSVs |
| `model.max_sequence_length` | Input sequence length (default: 800) |
| `model.input_dim` | Number of input features (default: 4) |
| `model.num_classes` | Classification classes (default: 2) |
| `raspberry_pi.model_path` | Path to deployed model on Pi |
| `hardware.sampling_rate` | Sensor sampling rate (default: 100Hz) |

### config.local.yaml

Create this file (gitignored) to override base settings for your environment. Only include values that differ from the base config.

## Troubleshooting

### I2C Device Not Found

```bash
# Check if I2C is enabled
ls /dev/i2c*

# Scan for devices
i2cdetect -y 1
```

If devices aren't detected:
1. Check wiring connections
2. Ensure I2C is enabled in raspi-config
3. Verify pull-up resistors on SDA/SCL lines

### Model Loading Errors

- Ensure PyTorch version matches between training and deployment
- Verify model file path in config or fall-detector.py
- Check available memory on Raspberry Pi

### Buzzer Not Sounding

- Verify GPIO 23 connection
- Check if GPIO is being used by another process
- Test with a simple GPIO script

### High CPU Usage

The fall detector runs continuously. On Raspberry Pi Zero 2W:
- CPU usage ~60-80% during inference is normal
- Consider reducing sampling rate if overheating occurs

## License

This project was developed as part of PhD research. Please contact the author for licensing information.

## Citation

If you use this work in your research, please cite:
```
[Citation information to be added]
```
