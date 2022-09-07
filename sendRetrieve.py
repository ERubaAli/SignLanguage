import requests

sTimeDate='05'

res=requests.get('http://127.0.0.1:5000/Retrieve_data',params={'table':"Sign",'sTimeDate':sTimeDate,'col':"sday"})

print(res.text)