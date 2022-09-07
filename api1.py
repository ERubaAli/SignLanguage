
from flask import Flask,  request
import mediapipe as mp
import pickle
import cv2
import numpy as np
import pandas as pd
import sqlite3
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


app = Flask(__name__)

conn = sqlite3.connect('Sign.db',check_same_thread=False)
cur = conn.cursor()

cred = credentials.Certificate("C:/Users/eRuba/Downloads/signwords-66f88-firebase-adminsdk-wmq6g-002a29ad96.json") 
firebase_admin.initialize_app(cred, {'databaseURL' : 'https://signwords-66f88-default-rtdb.firebaseio.com/' , 'httpTimeout' : 30})



@app.route('/predict', methods = [ 'POST','GET']) #prediction route
def pred():
    file=request.files['row']
    npd=pd.read_csv(file)

    with open('signL.pkl', 'rb') as f:
        model = pickle.load(f)
                    
    SignL_class=model.predict(npd)[0]
      
    return SignL_class



'''
@app.route("/Tcreate", methods=[ 'POST','GET'])       #create table
def CreateT():
   TName=request.args.get('TName')
   colDat=request.args.get('colDat')
   query1=f"CREATE TABLE {TName} ({colDat});"
   cur.execute(query1)
   conn.commit()
  
   return "DONE"

'''
# '''
@app.route("/inValues",methods=['POST','GET'])  #insert data to database
def insert():
    myTable=request.args.get('myTable')
    colm=request.args.get('colm')
    val=request.args.get('val')
    syear=request.args.get('syear')
    smonth=request.args.get('smonth')
    sday=request.args.get('sday')
    sTime=request.args.get('stime')
    sword=request.args.get('signW')
    query3=f'INSERT INTO {myTable} {colm} VALUES {val};'
        
    cur.execute(query3)
    conn.commit()
      
#saving to firebase

    ref = "Sign/"
    root = db.reference(ref)
    sign={"words":sword}
    root.child(syear).child(smonth).child(sday).child(sTime).update(sign)

    return "DONE"

# '''
'''
@app.route("/Retrieve_data")        #showing results
def getWords():
    table=request.args.get('table')
    sTimeDate=request.args.get('sTimeDate')
    col=request.args.get('col')
    queryS=f'SELECT * FROM {table} WHERE {col} = "{sTimeDate}";'
    cur.execute(queryS)
    Result=cur.fetchall()
    conn.commit()
    Res=pd.DataFrame([Result])
    Res.to_csv(f'D:\\dataScience\\SL\\retWords {sTimeDate}.csv',index=False)
    
    
    return f"words saved in retWords{sTimeDate}.csv"
'''


if __name__ == "__main__":
    app.run( debug=False)


conn.close()