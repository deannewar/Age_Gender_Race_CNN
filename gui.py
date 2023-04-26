import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

import cv2
import sys
import keras
import numpy as np
from PIL import Image
from tensorflow.keras.utils import load_img

screen = tk.Tk()
screen.geometry("1000x800")  
screen.tk.call('tk', 'scaling', 1.5)
screen.title('Age, Gender, Race Classification with CNN')
font1=('helvitica', 11)
font2=('helvitica', 8)
label_1 = tk.Label(screen,text='Upload an image with one or multiple faces \n for the prediction of the age, gender and \n ethnicity of the faces in the picture!',width=50,font=font1)  
label_1.grid(row=4,column=1,columnspan=3)
label_1.place(x=-40, y=80)
button_1 = tk.Button(screen, text='Upload Files', 
   width=15,command = lambda:upload_file())

button_1.grid(row=5,column=2,columnspan=3)
button_1.place(x=140, y=150)

def extract_features(images):
    features =[]
    img = load_img(images, grayscale=True)
    img = features.append(np.array(img.resize((100, 100), Image.ANTIALIAS)))
    features = np.array(features)
    return features

def run_model(image):
    image_extracted = extract_features(image)
    reconstructed_model_age = keras.models.load_model("./Age_Gender_Race_CNN/age_model.h5", compile=False)
    reconstructed_model_gender = keras.models.load_model("./Age_Gender_Race_CNN/gender_model.h5", compile=False) 
    reconstructed_model_race = keras.models.load_model("./Age_Gender_Race_CNN/race_model.h5", compile=False)
    
    age_pred = reconstructed_model_age.predict(image_extracted.reshape(1, 100, 100, 1))
    gender_pred = reconstructed_model_gender.predict(image_extracted.reshape(1, 100, 100, 1))
    race_pred = reconstructed_model_race.predict(image_extracted.reshape(1, 100, 100, 1))
    age_pred_prob, gender_pred_prob, race_pred_prob = age_pred, gender_pred, race_pred 
    age_pred, gender_pred, race_pred = round(np.amax(age_pred)), round(np.amax(gender_pred)), round(np.amax(race_pred))

    if gender_pred == 1:
        gender = 'Female'
    if gender_pred == 0:
        gender = 'Male'
    if gender_pred != 1 and gender_pred != 0:
        gender = 'Unknown Gender'
    
    if age_pred == 0:
        age = '<18'
    if age_pred == 1:
        age = '18-30'
    if age_pred == 2:
        age = '30-40'
    if age_pred == 3:
        age = '40-50'
    if age_pred == 4:
        age = '50-60'
    if age_pred == 5:
        age = '60>'
    
    if age_pred != 0 and age_pred != 1 and age_pred != 2 and age_pred != 3 and age_pred != 4 and age_pred != 5:
        age = 'Unknown Age' 
    
    if race_pred == 0:
        eth = 'White'
    if race_pred == 1:
        eth = 'Black'
    if race_pred == 2:
        eth = 'Asian'
    if race_pred == 3:
        eth = 'Indian'
    if race_pred == 4:
        eth = 'Others'
    
    if race_pred != 0 and race_pred != 1 and race_pred != 2 and race_pred != 3 and race_pred != 4:
        eth = 'Unknown Ethnicity'
    list_pred = [age, gender, eth]
    list_prob = [age_pred_prob, gender_pred_prob, race_pred_prob]
    return list_pred, list_prob

def upload_file():
    f_types = [('Jpg Files', '*.jpg'),
    ('PNG Files','*.png')]  
    filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)

    col=1 
    row=3 
    for f in filename:
        img = cv2.imread(f)
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)

        color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = faceCascade.detectMultiScale(
            color,
            scaleFactor=1.6,
            minNeighbors=5,
            minSize=(30, 30)
            )
        
        predictions = []
        probabilities = []
        padding = 10
        for (x, y, w, h) in faces:
            cv2.rectangle(img,(x-padding,y-padding),(x+w+padding,y+h+padding), (255, 0, 0), 1)
            roi_color = img[y:y + h, x:x + w]
            cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)
            im = f'./Age_Gender_Race_CNN/{str(w) + str(h)}_faces.jpg'

            preds, probs= run_model(im)
            predictions.append(preds)
            probabilities.append(probs)
            count = np.where(faces== x)[0][0]

            cv2.putText(img, f'Face {count}',  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
        cv2.imwrite('faces_detected.jpg', img)

        img=Image.open('faces_detected.jpg') 
        img.thumbnail((500, 500))
        img.save('thumbnail.jpg')
        print(img.size) 
        img2=Image.open('thumbnail.jpg')
        img2=ImageTk.PhotoImage(img2)
        pic_1 =tk.Label(screen)
        pic_1.grid(row=row,column=col)
        pic_1.place(x=400, y=100)
        pic_1.image = img2 
        pic_1['image']=img2 
        if(col==3): 
            row=row+1
            col=1    
        else:       
            col=col+1 
        
        prediction_text = ''
        for x in range(len(predictions)):
            prediction_text += f'Face {x} predictions: Age={predictions[x][0]}, Gender={predictions[x][1]}, Race={predictions[x][2]} \n '  

        label_2 = tk.Label(screen,text=prediction_text, width=100,font=font2)  
        label_2.grid(row=7,column=3,columnspan=3)
        label_2.place(x=350, y=450)

                           
screen.mainloop() 