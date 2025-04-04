import torch
import time
from dataset import get_dataloader
from model import EmotionModel

# Start timing
start_time = time.time()

print("Evaluation started...")

# Load test data
test_loader, class_names = get_dataloader("data/validation", batch_size=32, shuffle=False)
num_classes = len(class_names)

# Load trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionModel(num_classes).to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))  # Ensure correct device
model.eval()

# Evaluate accuracy
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

# End timing
end_time = time.time()
elapsed_time = end_time - start_time

# Print results
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Evaluation completed in {elapsed_time:.2f} seconds.")
