model file:
1- gets data to form the coordinates.csv file as the dataset
2- trains the model using Ridgeclassifier
3- save the model as signL.pkl


api1 file: (run this file )
1- route for prediction
2- route for creating the table in sqlite3
3- route for inserting values in database and firebase
4- route for retrieving data and saving them in a file with the name rewords and followed by the number inserted

send1 file:(then run this file )
1- turns the webcam on and draws the hand's landmarks and saves them to send them to the API to make the predictions
2- sends a request to create the table 
3- sends a request to insert data into database and firebase

sendRetrieve file: (you can run it to retrieve data after runnung the api file)
1- sends a request to retrieve data from database according to day

-----------------------------------------------------------------------------------
you need to:
1- open send1 and sendRetrieve in a new window
2- change the path of the row.csv file (written in the send1 file) to the path of your API
3- change the path ot the retwords file (written in the api1 file in Retrieve_data route) to where you want to save
your retrieved data
4- change the sTimeDate variable (written in sendRetrieve file) to the day you want the data to be retrieved as
a number between single quotations
