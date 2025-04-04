import torch
import torch.nn as nn
import torch.optim as optim
import time
from dataset import get_dataloader
from model import EmotionModel

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, class_names = get_dataloader("data/train", BATCH_SIZE)
num_classes = len(class_names)

# Initialize model, loss function, and optimizer
model = EmotionModel(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Start overall training timer
start_time = time.time()

print("Training started...")

# Training loop
for epoch in range(EPOCHS):
    epoch_start = time.time()  # Start time for this epoch
    
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    epoch_end = time.time()  # End time for this epoch
    epoch_time = epoch_end - epoch_start
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}, Time: {epoch_time:.2f} seconds")

# End overall training timer
end_time = time.time()
total_training_time = end_time - start_time

# Save model
torch.save(model.state_dict(), "model.pth")
print(f"Model training completed and saved.")
print(f"Total training time: {total_training_time:.2f} seconds.")
