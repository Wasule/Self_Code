import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os

# Detector model definition (same as in detect.py)
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

# Load trained detector
model_path = "detector.pth"
detector = Detector()
if os.path.exists(model_path):
    detector.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
else:
    # If no trained model exists, save the randomly initialized detector so the app can run.
    torch.save(detector.state_dict(), model_path)
detector.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Prediction function
def classify(img):
    # Ensure PIL image is RGB
    if isinstance(img, Image.Image) and img.mode != "RGB":
        img = img.convert("RGB")
    img = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = detector(img)
        pred = torch.argmax(output, dim=1).item()
    return "Adversarial" if pred == 1 else "Clean"

# Gradio interface
interface = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="Adversarial Attack Detector",
    description="Upload an image (32x32). The model will classify it as Clean or Adversarial."
)

if __name__ == "__main__":
    interface.launch()
