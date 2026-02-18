# Adversarial Attack Detection in Image Classification

This project demonstrates how to train a baseline CNN on CIFAR-10 and then build a detector to distinguish between clean images and adversarially perturbed images.

## Environment Setup

1. Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r adversarial_detection/requirements.txt
```

2. Files of interest:

- Baseline training: [adversarial_detection/train.py](adversarial_detection/train.py)
- Generate adversarial examples & train detector: [adversarial_detection/detect.py](adversarial_detection/detect.py)
- Adversarial generator & tester: [adversarial_detection/generate_attack.py](adversarial_detection/generate_attack.py)

## Quick Run

1. Train baseline model:

```bash
python3 adversarial_detection/train.py
```

This saves `baseline_cnn.pth`.

2. Train detector (generates adversarial examples and trains detector):

```bash
python3 adversarial_detection/detect.py
```

This saves `detector.pth`.

3. Generate a single adversarial example and run the detector:

```bash
python3 adversarial_detection/generate_attack.py --attack fgsm --eps 0.01
```

Output images are saved to `adv_output/` (clean and adversarial images). The script will also print the detector's prediction.

## Generate Multiple Adversarial Samples

To produce multiple adversarial examples with different perturbation sizes (`eps`) and save them into separate folders, run the following shell loop from the repository root:

```bash
for eps in 0.001 0.005 0.01 0.03; do
  outdir=adversarial_detection/adv_output/eps_${eps}
  python3 adversarial_detection/generate_attack.py --attack fgsm --eps ${eps} --outdir ${outdir}
done
```

After the loop completes you will have folders under `adversarial_detection/adv_output/` containing `clean.png` and `adversarial_fgsm_eps{eps}.png` for each `eps` value. Use these images to visually inspect attacks or to evaluate detector performance.

## Notes

- The project uses `torchattacks` for FGSM/PGD implementations. See `adversarial_detection/requirements.txt`.
- Models in the repo are simple and intended for demonstration; you can replace `baseline_cnn.pth` with your own stronger classifier if needed.

Feel free to open an issue or submit a PR with improvements.
