# Adversarial Detection Project

A comprehensive project for detecting adversarial attacks in image classification systems using deep learning.

## What is This Project About?

Adversarial attacks are carefully crafted perturbations added to images that can fool machine learning models into making incorrect predictions while remaining imperceptible to human eyes. This project addresses the critical problem of **detecting whether an image has been adversarially attacked** rather than trying to make models robust against attacks.

The system trains two complementary models:
1. **Baseline CNN** - A standard image classifier trained on clean CIFAR-10 images
2. **Adversarial Detector** - A specialized detector that learns to distinguish between clean and adversarially perturbed images

## Why Was This Developed?

### The Problem
- Deep learning models are vulnerable to adversarial attacks
- Adversarial perturbations are often imperceptible to humans but fool AI systems
- Defense mechanisms have limitations and can be computationally expensive
- There's a need for alternative approaches to handle adversarial robustness

### The Solution
Rather than making models inherently robust (which is difficult), this project implements a **detection-based defense** approach:
- Deploy a lightweight detector alongside the classifier
- Identify suspicious inputs before they reach the main model
- Provide an additional security layer in critical applications
- Enable safe fallback mechanisms when poisoned inputs are detected

## How Does It Work?

### Architecture Overview

```
Clean Images / Adversarial Images
            ↓
    Feature Extraction (CNN)
            ↓
    Adversarial Detector Network
            ↓
    Classification: [Clean / Adversarial]
```

### Process Flow

1. **Training Phase**
   - Train baseline CNN on clean CIFAR-10 images
   - Generate adversarial examples using FGSM/PGD attacks with varying epsilon values
   - Train detector network to differentiate between clean and attacked images

2. **Attack Generation**
   - FGSM (Fast Gradient Sign Method) - fast, single-step perturbation
   - PGD (Projected Gradient Descent) - stronger, iterative attack method
   - Epsilon values control perturbation magnitude (0.001 to 0.03 range)

3. **Detection Mechanism**
   - Detector analyzes image features and pixel patterns
   - Learns subtle differences between clean and adversarial examples
   - Outputs confidence scores for classification

### Key Components

- **Baseline Model**: Simple CNN architecture trained on clean images
- **Detector Model**: Neural network trained on mixed clean/adversarial dataset
- **Attack Generator**: Creates adversarial examples with configurable parameters
- **Evaluator**: Measures detection accuracy and false positive/negative rates

## Overview

This repository contains code for training baseline CNN models and building detectors to identify adversarially perturbed images.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train baseline model
python3 train.py

# Train detector
python3 detect.py

# Generate adversarial examples
python3 generate_attack.py --attack fgsm --eps 0.01
```

## Files

- `train.py` - Train baseline CNN model
- `detect.py` - Train adversarial detector
- `generate_attack.py` - Generate adversarial examples
- `evaluate_detector.py` - Evaluate detector performance
- `app.py` - Application interface

## Requirements

See `requirements.txt` for dependencies.
