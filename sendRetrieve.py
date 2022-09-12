import requests

sday='06'
smonth='09'


res=requests.get('http://127.0.0.1:5000/Retrieve_data',params={'table':"Sign",'sday':sday,'smonth':smonth,'col1':"sday",'col2':"smonth"})

print(res.text)