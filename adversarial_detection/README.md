# Adversarial Attack Detection in Image Classification

This project demonstrates how to train a baseline CNN on CIFAR-10 and then build a detector to distinguish between **clean images** and **adversarially perturbed images**.  
It is research-based, reproducible, and deployable — ideal for AI + Cybersecurity applications.

---

## 📦 Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/adversarial_detection.git
cd adversarial_detection

# Create Virtual Environment
'''
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

'''
# Install Dependencies

pip install -r requirements.txt

# How to Run

1. Train Baseline CNN

python train.py

'''
Trains a simple CNN on CIFAR-10.

Saves the model as baseline_cnn.pth.
'''

2. Train Adversarial Detector

python detect.py

'''
Loads the baseline CNN.

Generates adversarial examples using FGSM.

Trains a binary classifier to detect adversarial vs clean images.

Saves the detector as detector.pth.
'''

# 📖 Documentation
Dataset: CIFAR-10 + adversarial examples (FGSM).

Modeling Decision: CNN baseline + adversarial detector.

Evaluation: Accuracy, ROC curve.

Error Analysis: Misclassification on subtle perturbations.

Responsible AI: Improves AI safety by detecting malicious inputs.

#🏆 Contribution
'''
Feel free to fork, improve, and submit pull requests. This project is designed to be reproducible and extendable for further research in AI + Cybersecurity.
'''