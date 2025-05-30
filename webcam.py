import cv2
import numpy as np
import torch
from torchvision import transforms
import models
import helper_fns as hf

NUM_CLASS_NAMES = 7
mapper = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'sadness',
    5: 'surprise',
    6: 'neutral'
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])

loaded_model_3 = models.EmotionsModelV4(input_shape = 1, # only one color channel
                              hidden_units = 32,
                              output_shape=NUM_CLASS_NAMES).to(device)
loaded_model_3.load_state_dict(torch.load("saved_models/model_3_very_long_train.pth")) #note the path may be different
loaded_model_3.to(device)




cam = cv2.VideoCapture(0)

cv2.namedWindow("Emotion Recognition")

img_counter = 0

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

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
        cropped_img = cv2.resize(roi_gray, (48, 48))
        img_tensor = transform(cropped_img)
        pred_prob_img = hf.make_predictions(loaded_model_3, [img_tensor], device)
        pred_class_img = pred_prob_img.argmax(dim=1)
        max_index = pred_class_img[0].item()
        cv2.putText(frame, mapper[max_index], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
        # prediction = model.predict(cropped_img)
        # maxindex = int(np.argmax(prediction))
        # cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        #             cv2.LINE_AA)

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

# made with help from: https://github.com/atulapra/Emotion-detection/tree/master?tab=readme-ov-file