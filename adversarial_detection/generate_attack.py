import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchattacks
from PIL import Image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.fc1 = nn.Linear(32*30*30, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 32*30*30)
        x = self.fc1(x)
        return x


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_baseline(path):
    net = Net()
    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        print(f"Loaded baseline model from {path}")
    else:
        print(f"Baseline model '{path}' not found. Using randomly initialized model.")
    net.eval()
    return net


def load_detector(path):
    det = Detector()
    if os.path.exists(path):
        det.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        print(f"Loaded detector model from {path}")
    else:
        print(f"Detector model '{path}' not found. Using randomly initialized detector.")
    det.eval()
    return det


def tensor_to_pil(tensor):
    # tensor expected in [C,H,W], range [0,1]
    transform = transforms.ToPILImage()
    return transform(tensor.cpu())


def main(args):
    transform = transforms.Compose([transforms.ToTensor()])

    if args.image is None:
        # use CIFAR10 test sample
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        img_tensor, label = testset[args.index]
        img = tensor_to_pil(img_tensor)
        inputs = img_tensor.unsqueeze(0)
        print(f"Using CIFAR10 test index {args.index} (label {label})")
    else:
        # load image from path and resize to 32x32
        img = Image.open(args.image).convert('RGB')
        img = img.resize((32, 32))
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(img)
        inputs = img_tensor.unsqueeze(0)
        print(f"Loaded image from {args.image}")

    # Load models
    baseline = load_baseline(args.baseline)
    detector = load_detector(args.detector)

    # Create attack
    atk_name = args.attack.lower()
    if atk_name == 'fgsm':
        atk = torchattacks.FGSM(baseline, eps=args.eps)
    elif atk_name == 'pgd':
        atk = torchattacks.PGD(baseline, eps=args.eps, alpha=args.alpha, steps=args.steps)
    else:
        raise ValueError('Unsupported attack: ' + args.attack)

    # Prepare fake labels for untargeted attack (use baseline prediction)
    with torch.no_grad():
        preds = baseline(inputs)
        labels = preds.argmax(dim=1)

    adv = atk(inputs, labels)

    # Save images
    os.makedirs(args.outdir, exist_ok=True)
    clean_path = os.path.join(args.outdir, 'clean.png')
    adv_path = os.path.join(args.outdir, f'adversarial_{atk_name}_eps{args.eps}.png')

    tensor_to_pil(inputs.squeeze(0)).save(clean_path)
    tensor_to_pil(adv.squeeze(0)).save(adv_path)
    print(f"Saved clean image to {clean_path}")
    print(f"Saved adversarial image to {adv_path}")

    # Run detector
    to_tensor = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    det_in = to_tensor(Image.open(adv_path)).unsqueeze(0)
    with torch.no_grad():
        out = detector(det_in)
        pred = out.argmax(dim=1).item()
    label_str = 'Adversarial' if pred == 1 else 'Clean'
    print(f"Detector prediction on adversarial image: {label_str} (class {pred})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate adversarial image and test detector')
    parser.add_argument('--image', type=str, default=None, help='Path to input image (optional)')
    parser.add_argument('--index', type=int, default=0, help='CIFAR10 test index to use if no image provided')
    parser.add_argument('--baseline', type=str, default='baseline_cnn.pth', help='Path to baseline model')
    parser.add_argument('--detector', type=str, default='detector.pth', help='Path to detector model')
    parser.add_argument('--attack', type=str, default='fgsm', choices=['fgsm','pgd'], help='Attack type')
    parser.add_argument('--eps', type=float, default=0.007, help='Epsilon for FGSM/PGD')
    parser.add_argument('--alpha', type=float, default=0.001, help='Alpha for PGD')
    parser.add_argument('--steps', type=int, default=10, help='Steps for PGD')
    parser.add_argument('--outdir', type=str, default='adv_output', help='Output directory for images')
    args = parser.parse_args()
    main(args)
