import torch
import torchvision.transforms as transforms
from PIL import Image
from model import EmotionModel

# Load class labels
class_names = ["angry", "happy", "neutral", "sad", "surprised"]

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionModel(num_classes=len(class_names)).to(DEVICE)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Function to predict emotion
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return class_names[predicted.item()]

# Test prediction
if __name__ == "__main__":
    image_path = input("Enter image path: ")  # Ask user for image path
    print(f"Predicted Emotion: {predict(image_path)}")
