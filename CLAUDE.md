# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fall detection system combining IoT sensors (Raspberry Pi + MPU6050 accelerometer) with transformer-based neural networks for real-time fall classification. The system collects accelerometer data at 100Hz, processes it through trained models, and triggers alerts via GPIO-connected buzzer.

## Key Commands

### Raspberry Pi Data Collection
```bash
# Collect fall data
python3 iot/pi/data-collector-falls.py

# Collect ADL (Activities of Daily Living) data
python3 iot/pi/data-collector-adl.py

# Run real-time fall detection
python3 iot/pi/fall-detector.py
```

### Raspberry Pi Setup
```bash
sudo apt install -y i2c-tools python3-pip
pip install pandas smbus numpy scikit-learn board Adafruit-Blinka adafruit-circuitpython-bmp3xx --break-system-packages
```

### Training Models
Run variation-specific training scripts from `training/variations/`:
```bash
python3 training/variations/transformer/transformer-model.py
python3 training/variations/informer/informer-model.py
# etc.
```

### TFLite Conversion (for embedded deployment)
```bash
python3 iot/arduino/converter/convert_to_tf_lite.py
```

## Architecture

### Data Pipeline
- **Sensor sampling**: 100Hz (10ms intervals)
- **Input channels**: AccX_filtered, AccY_filtered, Corrected_AccZ, Acc_magnitude (4 dimensions)
- **Sequence length**: 800 timesteps (padded/truncated)
- **Normalization**: StandardScaler per-sample
- **Classification**: Binary (Fall vs Non-Fall/ADL)

### Model Architectures (training/models/)
Multiple transformer variants implemented for comparison:
- `transformer.py` - Base transformer
- `informer.py` - Sparse attention (ProbSparse)
- `linformer.py` - Linear complexity attention
- `performer.py` - Kernel-based efficient attention
- `multi_scale_transformer.py` - Multi-resolution processing
- `temporal_convolutional_transformer.py` - TCN + transformer hybrid
- `lstm_transformer.py` - LSTM-transformer hybrid
- `t2v_bert.py` - Time2Vec + BERT

### Project Structure
- `iot/pi/` - Raspberry Pi data collectors and real-time detector
- `training/models/` - Model architecture definitions
- `training/variations/` - Training scripts with hyperparameter configs per model
- `training/utils/` - Dataset loading, training loops, constants
- `dataset/` - Data processing and visualization tools
- `3d-models/` - OpenSCAD enclosure designs
- `pcb/` - Circuit board designs

### Key Constants (training/utils/constants.py)
```python
max_sequence_length = 800
input_dim = 4
num_classes = 2
csv_columns = ['AccX_filtered_i', 'AccY_filtered_i', 'Corrected_AccZ_i', 'Acc_magnitude_i']
```

### Hardware Configuration
- MPU6050 accelerometer at I2C address 0x68
- BMP388 barometric sensor
- GPIO pins 23, 24, 25 for controls
- Piezo buzzer for fall alerts

## Hyperparameter Tuning

Uses Optuna for automated hyperparameter search. Tuning scripts follow pattern:
`training/variations/<model>/transformer-hyperparamter-tuning.py`

Constraints: model size must be < 5MB for embedded deployment.
