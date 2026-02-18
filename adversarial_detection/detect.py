import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchattacks

# Load baseline model
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

net = Net()
net.load_state_dict(torch.load("baseline_cnn.pth"))
net.eval()

# Dataset
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Generate adversarial examples
atk = torchattacks.FGSM(net, eps=0.007)

adv_data, adv_labels, clean_data, clean_labels = [], [], [], []
for inputs, labels in testloader:
    adv = atk(inputs, labels)
    adv_data.append(adv)
    adv_labels.append(torch.ones(len(labels)))  # adversarial = 1
    clean_data.append(inputs)
    clean_labels.append(torch.zeros(len(labels)))  # clean = 0

X = torch.cat(clean_data + adv_data)
y = torch.cat(clean_labels + adv_labels)

# Detection model
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

detector = Detector()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(detector.parameters(), lr=0.001)

# Train detector
for epoch in range(5):
    optimizer.zero_grad()
    outputs = detector(X)
    loss = criterion(outputs, y.long())
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

torch.save(detector.state_dict(), "detector.pth")
print("Detector model saved as detector.pth")
