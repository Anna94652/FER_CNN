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
import test as t

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
                             data_loader = train_dataloader,
                             loss_fn = loss_fn,
                             accuracy_fn = accuracy_fn,
                             device = device)
print(loaded_model_3_results)
