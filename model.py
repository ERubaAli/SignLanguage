import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp
import time
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle




#utilties to help drawing landmarks and connections on the hand
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

'''
#prepare the first row to be in the form 
# class, x1,y1,z1,x2,y2,z2,.......x21,y21,z21
num_coord=21
landmarks=['class']
for val in range(1,num_coord+1):
    landmarks += ['x{}'.format(val),'y{}'.format(val),'z{}'.format(val)]

#saving it to coordinates.csv file using the writerow method
with open('coordinates.csv',mode='w',newline='') as coordF:
    csv_writer=csv.writer(coordF,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks) 

'''
#get data
#collect landmarks and connections for each sign word and save the rows
#in coordinates.csv file
'''
class_name='Yes'
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      

    
    image.flags.writeable = False
    #change color mode to RGB so media pipe can deal with it
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    #change color mode back to  BGR so opencv can deal with it
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #get the landmarks
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    
    #export landmarks of classes
    lmList=[]
    try:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate (hand_landmarks.landmark):
                 h,w,c=image.shape
                 cx, cy,cz= lm.x, lm.y, lm.z
                 lmList.append([cx,cy,cz])
        #flatten the np array of the array then change it back to list so
        # we can export it as csv        
        row=list(np.array(lmList).flatten())
        #append class name
        row.insert(0, class_name)

        #save collected data to coordinates.csv 
        with open('coordinates.csv',mode='a',newline='') as coordF:
            csv_writer=csv.writer(coordF,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row) 
    

    except:
        pass

    # Flip the image horizontally
    cv2.imshow('MP Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
# '''
#train
'''
#reading the file into a dataframe
df=pd.read_csv('coordinates.csv')
#Drop columns which contain missing value
df=df.dropna(axis=1)

#Drop column class to keep x as input data(features)
x=df.drop('class',axis=1)
#save y as output which is the class column (target value)
y=df['class']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=200)

#preprocessing function to standarize data
#subtracts the mean and divides it by standard deviation
#the model will not be biased to certain features more than others
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#the model chosen after trying other models 
model =RidgeClassifier()
model.fit(X_train, y_train)

#predicting using xtest and getting the accuracy score
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print('The Accuracy for Test Set is {}'.format(test_acc*100))

#saving the model after training as signL.pkl
with open('signL.pkl','wb')as f:
    pickle.dump(model,f)

'''







