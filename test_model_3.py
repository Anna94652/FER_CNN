import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchmetrics import Accuracy
from torchmetrics import ConfusionMatrix
import mlxtend
from mlxtend.plotting import plot_confusion_matrix
from tqdm.auto import tqdm
import helper_fns as hf
import models
from pathlib import Path
import test as t
import matplotlib.pyplot as plt
import random
from PIL import Image


mapper = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'sadness',
    5: 'surprise',
    6: 'neutral'
}
class_names = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
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
NUM_CLASS_NAMES = 7
BATCH_SIZE = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Turn datasets into iterables (batches)
train_dataloader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE,
                              shuffle = True) # will shuffle the training data
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle = False)
accuracy_fn = Accuracy(task="multiclass", num_classes=7).to(device)
loss_fn = nn.CrossEntropyLoss()
loaded_model_3 = t.EmotionsModelV1(input_shape = 1, # only one color channel
                              hidden_units = 32,
                              output_shape=NUM_CLASS_NAMES).to(device)
loaded_model_3.load_state_dict(torch.load("saved_models/model_3_very_long_train.pth")) #note the path may be different
loaded_model_3.to(device) 
loaded_model_3_results = hf.eval_model(model=loaded_model_3,
                             data_loader = test_dataloader,
                             loss_fn = loss_fn,
                             accuracy_fn = accuracy_fn,
                             device = device)
print(loaded_model_3_results)
targets = []
for X, y in test_data:
    targets.append(y)
y_preds = []
targets = torch.tensor(targets)
loaded_model_3.eval()
with torch.inference_mode():
  for X, y in test_dataloader:
    # Send the data and targets to target device
    X, y = X.to(device), y.to(device)
    # Do the forward pass
    y_logit = loaded_model_3(X)
    # Turn predictions from logits -> Prediction probabilities -> prediction labels
    y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)
    # Put prediction on CPU for evaluation
    y_preds.append(y_pred.cpu())

# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)
confmat = ConfusionMatrix(num_classes = NUM_CLASS_NAMES, task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target = targets)

# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat = confmat_tensor.numpy(),
    class_names  = class_names,
    figsize = (10,7)
)
plt.show()

# Visualize images and their predicted labels
# 1. Get random samples of images
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
  test_samples.append(sample)
  test_labels.append(label)
pred_probs = hf.make_predictions(model=loaded_model_3,
                              data=test_samples)
pred_classes = pred_probs.argmax(dim=1)

# Plot predictions
plt.figure(figsize= (9,9))
nrows=3
ncols=3
for i, sample in enumerate(test_samples):
  # Create subplot
  plt.subplot(nrows, ncols, i+1)

  # plot the target image
  plt.imshow(sample.squeeze(), cmap="gray")

  # Find the prediction 
  pred_label = class_names[pred_classes[i]]

  # Get the truth label (in text form)
  truth_label = class_names[test_labels[i]]

  # Create a title for the plot3
  title_text = f"Pred: {pred_label} | Truth: {truth_label}"

  # Check for equality between pred and truth and change color of title text
  if pred_label == truth_label:
    plt.title(title_text, fontsize=10, c="g") # green text if prediction same as truth
  else:
    plt.title(title_text, fontsize=10, c="r")

  plt.axis(False)

plt.show()

# Visualize images and their predictions on our own images
# 1. Get image into a tensor
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()]
)
image_happy = Image.open("images/happy.png")
image_happy_tensor = transform(image_happy)
happy_list = [image_happy_tensor]
pred_prob_image_happy = hf.make_predictions(loaded_model_3, happy_list, device)
pred_class_image_happy = pred_prob_image_happy.argmax(dim=1)
plt.figure(figsize= (9,9))
plt.imshow(image_happy_tensor.squeeze(), cmap = "gray")
print(pred_class_image_happy[0].item())
pred_label_image_happy = class_names[pred_class_image_happy[0].item()]
title_text = f"Pred: {pred_label_image_happy}"
plt.title(title_text)
plt.axis(False)
plt.show()
