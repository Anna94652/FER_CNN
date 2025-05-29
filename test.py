import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import pandas as pd
import torchmetrics
from torchmetrics import Accuracy
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
# Getting FER2013 dataset
# Setup training data
train_transform_trivial_augment = transforms.Compose([
    transforms.TrivialAugmentWide(num_magnitude_bins=5),
    transforms.ToTensor() 
])
train_data = datasets.FER2013(
    root = "data",
    split = "train",
    transform = train_transform_trivial_augment,
    target_transform = None
)
# Setup test data
test_data = datasets.FER2013(
    root = "data",
    split = "test",
    transform = ToTensor(),
    target_transform = None
)

BATCH_SIZE = 32
NUM_CLASS_NAMES = 7
# Turn datasets into iterables (batches)
train_dataloader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE,
                              shuffle = True) # will shuffle the training data
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle = False)
accuracy_fn = Accuracy(task="multiclass", num_classes=7).to(device)
loss_fn = nn.CrossEntropyLoss()

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
  """ Performs a training with model trying to learn on data_loader."""
  train_loss, train_acc = 0, 0
  # Put model into training mode
  model.train()
  # Add a loop to loop through the training batches
  for batch, (X, y) in enumerate(data_loader):
    # Put data on target device
    X, y = X.to(device), y.to(device)
    # 1. Forward pass (outputs the raw logits from the model)
    y_pred = model(X)

    # 2. Calculate loss and accuracy (per batch)
    loss = loss_fn(y_pred, y)
    train_loss += loss # accumulate train loss
    train_acc += accuracy_fn(y_pred.argmax(dim=1), y) # go from logits -> prediction labels

    # 3. Optimizer zero grad
    optimizer.zero_grad()
    # 4. Loss backward
    loss.backward()
    # 5. Optimizer step
    optimizer.step()

  # Divide total train loss and accuracy by length of train dataloader
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device:torch.device = device):
  """ Performs a testing loop step on model going over data_loader."""
  test_loss, test_acc = 0,0
  # Put the model in eval mode
  model.eval()
  # Turn on inference mode context manager
  with torch.inference_mode():
    for X, y in data_loader:
      # Send the data to the target device
      X, y = X.to(device), y.to(device)
      # 1. Forward pass
      test_pred = model(X)
      # 2. Calculate the loss/acc
      test_loss += loss_fn(test_pred, y)
      test_acc += accuracy_fn(test_pred.argmax(dim=1), y) # go from logits -> prediction labels
    # Adjust metrics and print out
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")

class EmotionsModelV0(nn.Module):
  def __init__(self,
               input_shape: int,
               hidden_units: int,
               output_shape: int):
    super().__init__()
    self.layer_stack = nn.Sequential(
        nn.Flatten(), # flatten layer will reduce shape of the input tensor
        nn.Linear(in_features=input_shape,
                  out_features=hidden_units),
        nn.Linear(in_features=hidden_units,
                  out_features=output_shape)
    )
  def forward(self, x):
    return self.layer_stack(x)
  
torch.manual_seed(42)

# Setup model with input parameters
model_0 = EmotionsModelV0(
  input_shape = 2304, # the output of flatten needs to be the input shape (28*28)
  hidden_units = 10, # how many units in the hidden layer
  output_shape = 7 # one for every class
).to("cpu")

torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device = device):
    """ Returns a dictionary containing the seults of model predicting on data_loader. """
    loss, acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
          # Make our data device agnostic
          X, y = X.to(device), y.to(device)
          # Make predictions
          y_pred = model(X)

          # Accumulate the loss and acc values per batch
          loss += loss_fn(y_pred, y)
          acc += accuracy_fn(y_pred.argmax(dim=1), y)
        # Scale the loss and acc to find the average loss/acc per batch
        loss/=len(data_loader)
        acc/=len(data_loader)
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc.item()}

model_0_results = eval_model(model=model_0,
                             data_loader = test_dataloader,
                             loss_fn = loss_fn,
                             accuracy_fn = accuracy_fn,
                             device = "cpu")

# Create a convolutional neural network
class EmotionsModelV1(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()
    self.layer_stack = nn.Sequential(
        # block 1:
        nn.Conv2d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
        nn.BatchNorm2d(hidden_units),
        nn.PReLU(),
        nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
        nn.BatchNorm2d(hidden_units),
        nn.PReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25), # helps with overfitting
        # block 2:
        nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units*2, #increasing hidden units here
                    kernel_size=3,
                    stride=1,
                    padding=1),
        nn.BatchNorm2d(hidden_units*2),
        nn.PReLU(),
        nn.Conv2d(in_channels=hidden_units*2,
                    out_channels=hidden_units*2,
                    kernel_size=3,
                    stride=1,
                    padding=1),
        nn.BatchNorm2d(hidden_units*2),
        nn.PReLU(),
        nn.MaxPool2d(kernel_size=2), 
        nn.Dropout(0.25),
        #block 3:
        nn.Conv2d(in_channels=hidden_units * 2,
                    out_channels=hidden_units*4,
                    kernel_size=3,
                    stride=1,
                    padding=1),
        nn.BatchNorm2d(hidden_units*4),
        nn.PReLU(),
        nn.Conv2d(in_channels=hidden_units*4,
                    out_channels=hidden_units*4,
                    kernel_size=3,
                    stride=1,
                    padding=1),
        nn.BatchNorm2d(hidden_units*4),
        nn.PReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25),
        # block 4:
        nn.Conv2d(in_channels=hidden_units * 4,
                    out_channels=hidden_units*8,
                    kernel_size=3,
                    stride=1,
                    padding=1),
        nn.BatchNorm2d(hidden_units*8),
        nn.PReLU(),
        nn.Conv2d(in_channels=hidden_units*8,
                    out_channels=hidden_units*8,
                    kernel_size=3,
                    stride=1,
                    padding=1),
        nn.BatchNorm2d(hidden_units*8),
        nn.PReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25),
        # classifier:
        nn.Flatten(),
        nn.Linear(in_features = hidden_units * 72,
            out_features = 256),
        nn.PReLU(), 
        nn.Dropout(0.5),
        nn.Linear(in_features = 256, out_features = output_shape) #added another linear layer here
        )
  def forward(self, x):
      return self.layer_stack(x)

model_1 = EmotionsModelV1(input_shape = 1, # only one color channel
                              hidden_units = 32,
                              output_shape=NUM_CLASS_NAMES).to(device)

optimizer_model_1 = torch.optim.SGD(params=model_1.parameters(), lr=0.01)
loss_fn_model_1 = nn.CrossEntropyLoss()

# training model_1
torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 100
for epoch in range(epochs):
  print(f"Epoch: {epoch}\n--------")
  train_step(model=model_1,
              data_loader = train_dataloader,
              loss_fn = loss_fn_model_1,
              optimizer = optimizer_model_1,
              accuracy_fn = accuracy_fn,
              device = device
              )
  test_step(model=model_1,
              data_loader = test_dataloader,
              loss_fn = loss_fn_model_1,
              accuracy_fn=accuracy_fn,
              device = device)
  
model_1_results = eval_model(model=model_1,
                             data_loader = test_dataloader,
                             loss_fn = loss_fn_model_1,
                             accuracy_fn = accuracy_fn,
                             device = device)
print(model_1_results)

# Save model
Path("saved_models").mkdir(parents=True, exist_ok=True)
torch.save(model_1.state_dict(), "saved_models\\model_3_very_long_train.pth")