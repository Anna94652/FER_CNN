import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchmetrics import Accuracy
from tqdm.auto import tqdm
import helper_fns as hf
import models
from pathlib import Path

# Constants
EPOCHS = 20
NUM_CLASS_NAMES = 7
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# Device agnostics
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device\n"
      f"Batch size: {BATCH_SIZE}\n"
      f"Epochs: {EPOCHS}\n"
      f"Learning rate: {LEARNING_RATE}\n"
      f"Number of classes: {NUM_CLASS_NAMES}\n")

### Set up data
# Getting FER2013 dataset
train_data = datasets.FER2013(
    root = "data",
    split = "train",
    transform = ToTensor(),
    target_transform = None
)
# Setup test data
test_data = datasets.FER2013(
    root = "data",
    split = "test",
    transform = ToTensor(),
    target_transform = None
)

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE,
                              shuffle = True) # will shuffle the training data
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle = False)

# Instantiate model
model_3 = models.EmotionsModelV3(input_shape = 1, # only one color channel
                              hidden_units = 10,
                              output_shape=NUM_CLASS_NAMES).to(device)

# Set up loss, optimizer, and accuracy functions
optimizer = torch.optim.SGD(params=model_3.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()
accuracy_fn = Accuracy(task="multiclass", num_classes=7).to(device)

# Training model
torch.manual_seed(42)
torch.cuda.manual_seed(42)

for epoch in tqdm(range(EPOCHS)):
  print(f"Epoch: {epoch}\n--------")
  hf.train_step(model=model_3,
             data_loader = train_dataloader,
             loss_fn = loss_fn,
             optimizer = optimizer,
             accuracy_fn = accuracy_fn,
             device = device
             )
  hf.test_step(model=model_3,
            data_loader = test_dataloader,
            loss_fn = loss_fn,
            accuracy_fn=accuracy_fn,
            device = device)


# Evaluate model
model_3_results = hf.eval_model(model=model_3,
                             data_loader = test_dataloader,
                             loss_fn = loss_fn,
                             accuracy_fn = accuracy_fn,
                             device = device)

print(model_3_results)

# Save model
Path("saved_models").mkdir(parents=True, exist_ok=True)
torch.save(model_3.state_dict(), "saved_models\\model_3.pth")