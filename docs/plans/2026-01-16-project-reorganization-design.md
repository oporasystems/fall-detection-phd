# Fall Detection Project Reorganization Design

**Date:** 2026-01-16
**Status:** Approved
**Goal:** Archive cleanup, production focus, and team-ready GitHub release

## Context

This is a completed PhD project for transformer-based fall detection using Raspberry Pi and MPU6050 accelerometer. The **Performer** model was selected as the best-performing architecture and is deployed in production (`iot/pi/fall-detector.py`).

The project needs cleanup for:
- Academic archival (thesis/publication ready)
- Production deployment (Raspberry Pi)
- Team collaboration (GitHub release)

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Chosen model | Performer | Best performance, already in production |
| Other model variants | Keep all | Preserve research history for reproducibility |
| IoT script refactoring | No | No tests, working code, risk of breaking |
| Path configuration | Config file (base + local override) | Team-friendly, explicit |
| Documentation | Full README | Comprehensive for onboarding |
| Reorganization scope | Quick wins only | Minimal disruption |

## Implementation Plan

### 1. Quick Win Fixes

#### 1.1 Remove duplicate code
- `training/models/multi_scale_transformer.py`: Delete duplicate `PositionalEncoding` class (lines 27-45)
- `training/utils/train_utils.py`: Remove duplicate `train_model()` function (lines 95-129)

#### 1.2 Update `.gitignore`
Add:
```
venv/
*.zip
.DS_Store
__pycache__/
*.pyc
config.local.yaml
```

#### 1.3 File naming
- Keep existing names as-is (no renaming)
- Document convention for future files: use underscores for Python files

### 2. Config File System

#### 2.1 Create `config.yaml` (committed)
Base config with defaults:
```yaml
# Base configuration - override in config.local.yaml
dataset:
  fall_folder: "./data/Fall"
  non_fall_folder: "./data/ADL"

raspberry_pi:
  model_path: "/home/ivanursul/performer_model.pt"
  offsets_path: "/home/ivanursul/mpu_offsets.json"
  data_output_path: "/home/ivanursul/accelerometer_data_raw"
```

#### 2.2 Create `config.local.yaml.example`
Template for personal overrides:
```yaml
# Copy to config.local.yaml and customize
dataset:
  fall_folder: "/your/path/to/Fall"
  non_fall_folder: "/your/path/to/ADL"
```

#### 2.3 Update `training/utils/constants.py`
Load base config, merge local overrides if present.

### 3. Documentation (README.md)

Structure:
1. Overview - Project description, key features
2. Hardware Requirements - Pi Zero 2W, MPU6050, BMP388, buzzer, buttons
3. Quick Start - Installation, systemd setup, running fall-detector.py
4. Project Structure - Directory tree with descriptions
5. Model Architectures - All 10 variants, why Performer was selected
6. Training Pipeline - Dataset format, training, hyperparameter tuning
7. Data Collection - ADL and falls scripts, heatmap system
8. Configuration - config.yaml vs config.local.yaml
9. Troubleshooting - Common issues

## What Stays Unchanged

- `iot/pi/data-collector-adl.py` - no refactoring
- `iot/pi/data-collector-falls.py` - no refactoring
- `iot/pi/fall-detector.py` - no refactoring
- Folder structure - no reorganization
- All 10 model variants - preserved as-is
- Existing file names - no renaming
- Training pipeline logic - unchanged

## Implementation Order

1. Fix `.gitignore` (prevents committing junk)
2. Remove duplicate code (quick fixes)
3. Add config file system
4. Write comprehensive README

## Deliverables

- Clean codebase with no duplicate code
- Config-based path management for team use
- Comprehensive README for onboarding
- All 10 model variants preserved with documentation
