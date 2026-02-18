import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchattacks


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


def load_model(path, model_cls):
    model = model_cls()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        print(f"Loaded {path}")
    else:
        print(f"Warning: {path} not found. Using untrained model.")
    model.eval()
    return model


def evaluate(baseline_path, detector_path, eps_list, batch_size, max_batches):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    baseline = load_model(baseline_path, Net)
    detector = load_model(detector_path, Detector)

    results = []
    for eps in eps_list:
        atk = torchattacks.FGSM(baseline, eps=eps)

        total_clean = 0
        fp_clean = 0
        total_adv = 0
        tp_adv = 0

        for i, (inputs, labels) in enumerate(testloader):
            if max_batches and i >= max_batches:
                break

            # torchattacks requires inputs to require grad for white-box attacks
            inputs_req = inputs.clone().detach().requires_grad_(True)
            adv = atk(inputs_req, labels)

            with torch.no_grad():
                out_clean = detector(inputs)
                pred_clean = out_clean.argmax(dim=1)

                out_adv = detector(adv)
                pred_adv = out_adv.argmax(dim=1)

            total_clean += inputs.size(0)
            fp_clean += (pred_clean == 1).sum().item()  # clean predicted adversarial = false positive

            total_adv += adv.size(0)
            tp_adv += (pred_adv == 1).sum().item()  # adversarial detected = true positive

        fpr = fp_clean / total_clean if total_clean else 0.0
        tpr = tp_adv / total_adv if total_adv else 0.0
        results.append((eps, total_clean, fp_clean, total_adv, tp_adv, fpr, tpr))

        print(f"eps={eps}: clean_FP {fp_clean}/{total_clean} (FPR={fpr:.3f}), adv_TP {tp_adv}/{total_adv} (TPR={tpr:.3f})")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate adversarial detector across epsilons')
    parser.add_argument('--baseline', default='baseline_cnn.pth')
    parser.add_argument('--detector', default='detector.pth')
    parser.add_argument('--eps', nargs='+', type=float, default=[0.001, 0.005, 0.01, 0.03])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-batches', type=int, default=50, help='Max number of batches to evaluate (use 0 for all)')
    args = parser.parse_args()

    max_batches = None if args.max_batches == 0 else args.max_batches
    results = evaluate(args.baseline, args.detector, args.eps, args.batch_size, max_batches)

    out_path = 'adversarial_detection/eval_results.txt'
    with open(out_path, 'w') as f:
        f.write('eps,total_clean,fp_clean,total_adv,tp_adv,fpr,tpr\n')
        for row in results:
            f.write(','.join(map(str, row)) + '\n')

    print(f"Saved summary to {out_path}")
