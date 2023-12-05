import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18
import numpy as np
import time

# Define the modified model architecture
class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.resnet = resnet18(weights=None)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

#set the device for the model to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load the saved model from its file location
model_path = "c:/Users/c000832213/Documents/DeanStuff/snek/Pill Identifier/dataset/pill_classifier.pth"
saved_state_dict = torch.load(model_path)

#create the model using resnet and move it to the device
model = CustomResNet(num_classes=2)
model.to(device)

#loads the current save state into the model and puts it into evaluation mode
model.load_state_dict(saved_state_dict, strict=False)
model.eval()

#set up the camera, accesing the default
cap = cv2.VideoCapture(0)

#define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#define the two classes pill and non pill
class_labels = ['non_pill', 'pill']

#track object variable, false on startup
track_object = False

#contours variable, empty array
contours = []

#the region for text, to be used to later place and mask it
text_region = [(0, 0), (350, 40)]  

#create the window
cv2.namedWindow("Pill Identification and Object Tracking")

while True:
    #capture camera frame
    ret, frame = cap.read()

    #image processing
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pil_image = transform(pil_image)
    input_image = pil_image.unsqueeze(0).to(device)

    #create mask to cover text region
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, text_region[0], text_region[1], (255), -1)

    #apply mask to the text frame and make it black
    frame[mask > 0] = [0, 0, 0]

    #convert frame to color frame
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #define lower and upper color range
    lower_color = np.array([40, 50, 50])
    upper_color = np.array([80, 255, 255])

    #create mask based on color image
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)

    #find contours in mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not track_object or len(contours) == 0:
        #check for contours
        if len(contours) > 0:
            #initialize the list to store the individual points
            points = []

            for contour in contours:
                #get contour points
                contour_points = cv2.convexHull(contour)

                #add contour points to list
                points.extend(contour_points)

            if len(points) > 0:
                #convert points to array
                points = np.array(points)

                #calculate points of rectangle
                rect = cv2.minAreaRect(points)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.intp)

                #draw the box
                if len(box) > 0:
                    cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                #and set track object to true
                track_object = True
            else:
                #track object if no points are found
                track_object = False
        else:
            #track object is false if no countours are found
            track_object = False
    else:
        #if track_object is True
        for contour in contours:
            contour_points = cv2.convexHull(contour)
            points = np.concatenate([points, contour_points])

        if len(points) > 0:
            points = np.array(points)
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.intp)
            if len(box) > 0:
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        else:
            #track object is false if no points are found
            track_object = False

    #run pill idenfitication
    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output, 1)
    if track_object:# and len(contours) > 0:
        #pill identification. Process and output results
        pill_label = class_labels[predicted.item()]
        #if a pill, say so and make text green
        if pill_label == 'pill':
            pill_text = 'Pill'
            pill_color = (0, 255, 0)
            track_object = True
        #if not a pill, say so and make text red
        if pill_label == 'non_pill':
            pill_text = 'Not a Pill'
            pill_color = (0, 0, 255)
            track_object = False        

    #if not tracking an object, say so and make text red
    if not track_object and len(contours) == 0:
        pill_text = 'No Object Detected'
        pill_color = (0,0,255)

    #add identification text to the frame
    cv2.putText(frame, pill_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, pill_color, 2)
    print(pill_text)

    #display the frame
    cv2.imshow("Pill Identification and Object Tracking", frame)

    #exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
