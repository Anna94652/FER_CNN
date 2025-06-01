import cv2
import tkinter as tk
from tkinter import filedialog
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
import models as models
from pathlib import Path
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageTk

################ START copy of needed code from test.py

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



class_names = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
img_counter = 0
NUM_CLASS_NAMES = 7
BATCH_SIZE = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



############### END



accuracy_fn = Accuracy(task="multiclass", num_classes=7).to(device)
loss_fn = nn.CrossEntropyLoss()
loaded_model_3 = EmotionsModelV1(input_shape = 1, # only one color channel
                              hidden_units = 32,
                              output_shape=NUM_CLASS_NAMES).to(device)
loaded_model_3.load_state_dict(torch.load("saved_models/model_3_very_long_train.pth"))
loaded_model_3.to(device) 
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()]
)
class FacialClassifier:
    def __init__(self, master):
        self.master = master
        master.title("Facial Emotion Classifier")
        self.master.geometry("400x400")

        self.upload_button = tk.Button(master, text="Upload an image of a face here", command= self.upload_image)
        self.upload_button.pack()

        self.camera_button = tk.Button(master, text="Or, use webcamera", command= self.camera)
        self.camera_button.pack()

        self.label = tk.Label(self.master)
        self.label.pack()

        self.predicted_emotion_text = tk.Label(master, text = "Predicted Emotion:")
        self.predicted_emotion_text.pack()

        self.prob_text_1 = tk.Label(master, text = "")
        self.prob_text_1.pack()
        self.prob_text_2 = tk.Label(master, text = "")
        self.prob_text_2.pack()
        self.prob_text_3 = tk.Label(master, text = "")
        self.prob_text_3.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(initialdir=".", title="Select an Image", filetypes=(("Image files", "*.png *.jpg *.jpeg"), ("all files", "*.*")))
        if file_path:
            img = Image.open(file_path)
            self.img_tensor = transform(img) #convert image to a tensor

            img = img.resize((250, 250)) #resize and show image
            photo = ImageTk.PhotoImage(img)
            self.label.config(image=photo)
            self.label.image = photo
            # Make prediction on the face's emotion
            list = [self.img_tensor]
            pred_prob = hf.make_predictions(loaded_model_3, list, device)
            # pred_class = pred_prob.argmax(dim=1)
            # pred_label = class_names[pred_class[0].item()]
            top_3 = torch.topk(pred_prob.flatten(), 3).indices
            pred_label = class_names[top_3[0]]
            pred_label_2 = class_names[top_3[1]]
            pred_label_3 = class_names[top_3[2]]
            # Change the text
            self.predicted_emotion_text.config(text = f"Predicted Emotion: {pred_label}")
            self.prob_text_1.config(text = f"{pred_label}: {pred_prob[0][top_3[0]]*100:.2f}%")
            self.prob_text_2.config(text = f"{pred_label_2}: {pred_prob[0][top_3[1]]*100:.2f}%")
            self.prob_text_3.config(text = f"{pred_label_3}: {pred_prob[0][top_3[2]]*100:.2f}%")
    def camera(self):
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Emotion Recognition")
        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)
        img_counter = 0

        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break


            # Find haar cascade to draw bounding box around face
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                # cropped_img = cv2.resize(roi_gray, (48, 48))
                img_tensor = transform(Image.fromarray(roi_gray))
                pred_prob_img = hf.make_predictions(loaded_model_3, [img_tensor], device)
                pred_class_img = pred_prob_img.argmax(dim=1)
                max_index = pred_class_img[0].item()
                cv2.putText(frame, class_names[max_index], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
                
            cv2.imshow("Emotion Recognition", frame)

            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1
        cam.release()
        cv2.destroyAllWindows()

        
root = tk.Tk()
emotion_classifier = FacialClassifier(root)
root.mainloop()