import requests
from time import  sleep
import mediapipe as mp
import cv2
import pandas as pd 
import numpy as np
import pickle
import time
from itertools import groupby
from datetime import datetime



#utilties to help drawing landmarks and connections on the hand
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

Sign_List=[]
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
                #  h,w,c=image.shape
                 cx, cy,cz= lm.x, lm.y, lm.z
                 lmList.append([cx,cy,cz])
        #flatten the np array of the array then change it back to list so
        # we can export it as csv
        row=list(np.array(lmList).flatten())

        #get dataframe from the row list to export it as csv file
        X_p=pd.DataFrame([row])
        #set index to false to get the number of features correctly
        X_p.to_csv('D:\\dataScience\\SL\\row.csv',index=False)
        pfile={'row':open('D:\\dataScience\\SL\\row.csv')}
        #send the file in a post request to the API to make the prediction 
        res=requests.post('http://127.0.0.1:5000/predict', files=pfile)
        SignL_class=res.text
        # get all the predicted words and append them to the Sign_List
        Sign_List.append(SignL_class)
        print(res)
        #showing the predicted word    
                    # Get status box
        cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                    
                    # Display Class
        cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, SignL_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # give some time to cancel predicting random hand movement between signs
        time.sleep(2)

    except:
         pass


    cv2.imshow('SignL',image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

print (Sign_List)  
#use group in groupby to eliminate the same words listed after each other
final_list=[key for key, _group in groupby(Sign_List)]  
print (final_list)
#get time without seconds and msec
cTime=datetime.now().replace(second=0, microsecond=0)
#join list elements and save them in a string
fListStr='/'.join(final_list)
print(fListStr)

stime=cTime.strftime('%H:%M')
sH=cTime.strftime('%H')
sM=cTime.strftime('%M')
sdate=cTime.strftime('%d/%m/%Y')
syear=cTime.strftime('%Y')
smonth=cTime.strftime('%m')
sday=cTime.strftime('%d')



#creating the table
# res=requests.post ('http://127.0.0.1:5000/Tcreate',params={'TName':'Sign','colDat':'Word  TEXT NOT NULL, wHour TEXT, wMin TEXT NOT NULL,  syear TEXT NOT NULL, smonth TEXT NOT NULL,sday TEXT NOT NULL'})

#inserting values to database and firebase
res=requests.post('http://127.0.0.1:5000/inValues',
                  params={'signW':f'{fListStr}','sH':f'{sH}' ,'sM':f'{sM}',
                  'syear':f'{syear}','smonth':f'{smonth}',
                  'sday':f'{sday}','stime':f'{stime}' })


print (res.text)