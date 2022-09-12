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
import urllib.request



app = Flask(__name__)

conn = sqlite3.connect('Sign.db',check_same_thread=False)
cur = conn.cursor()

cred = credentials.Certificate("C:/Users/eRuba/Downloads/signwords-66f88-firebase-adminsdk-wmq6g-002a29ad96.json") 
firebase_admin.initialize_app(cred, {'databaseURL' : 'https://signwords-66f88-default-rtdb.firebaseio.com/' , 'httpTimeout' : 30})


def getRec():
    queryCH='SELECT * FROM Sign WHERE flag = 0' 
    cur.execute(queryCH)
    rec=cur.fetchall()
    conn.commit()
    return rec


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
    # myTable=request.args.get('myTable')
    # colm=request.args.get('colm')
    # val=request.args.get('val')
        sword=request.args.get('signW')
        sH=request.args.get('sH')
        sM=request.args.get('sM')
        syear=request.args.get('syear')
        smonth=request.args.get('smonth')
        sday=request.args.get('sday')
        sTime=request.args.get('stime')
        queryIn=f'INSERT INTO Sign (Word,wHour,wMin,syear,smonth,sday) VALUES ("{sword}","{sH}","{sM}","{syear}","{smonth}","{sday}");'
        queryUp=f'UPDATE Sign SET flag = 1 WHERE flag=0'
        cur.execute(queryIn)
        conn.commit()
        try:
        
            Records=getRec()
            for row in Records:
                sword=row[0]
                sH=row[1]
                sM=row[2]
                syear=row[3]
                smonth=row[4]
                sday=row[5]
                dflag=row[6]
                
                #saving to firebase
                sTime=f'{sH}:{sM}'
                ref = "Sign/"
                root = db.reference(ref)
                sign={"words":sword}
                root.child(syear).child(smonth).child(sday).child(sTime).update(sign)
            cur.execute(queryUp)
            conn.commit()
            return "DONE"
        except:
            Records=getRec()
            return "connect"
        # syncDict={ref:{syear:{smonth:{sday:{sTime:sign}}}}}
        # print(syncDict)
            

# '''
'''
@app.route("/Retrieve_data")        #showing results
def getWords():
    table=request.args.get('table')
    sday=request.args.get('sday')
    smonth=request.args.get('smonth')
    col1=request.args.get('col1')
    col2=request.args.get('col2')
    queryS=f'SELECT * FROM {table} WHERE {col1} = "{sday}" AND {col2} = "{smonth}";'
    cur.execute(queryS)
    Result=cur.fetchall()
    conn.commit()
    Res=pd.DataFrame([Result])
    Res.to_csv(f'D:\\dataScience\\SL\\retWords {sday}-{smonth}.csv',index=False)
    
    
    return f"words saved in retWords{sday}-{smonth}.csv"
'''


if __name__ == "__main__":
    app.run( debug=False)


conn.close()