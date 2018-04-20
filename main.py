import numpy as np
import keras
#from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation
import requests
import json
import codecs

##### VARIABLES #####
num_to_average = 60 #Number of stock to compute the average


#### PREPROCESS DATA #####

# Function to get intraday trading data to be used as input data
def __get_input_data():
	# Query to get intraday data
	url = "https://www.alphavantage.co/query"
	querystring = {"apikey":"7CJAFX87TQG5584E","function":"TIME_SERIES_INTRADAY","symbol":"MSFT","interval":"1min","outputsize":"full"}
	headers = {
		'Cache-Control': "no-cache",
		'Postman-Token': "355d46af-af2c-4c27-9f92-f5bc46a4a9d8"
		}
	response_intraday_data = requests.request("GET", url, headers=headers, params=querystring)

	#Put input data in a numpy array array (n,4), where n=100 or ? (compact ot full).
	json_data = json.loads(response_intraday_data.text)
	aux2 = []
	for i,time_value in enumerate(json_data["Time Series (1min)"]):
		aux = []
		
		for j,value in enumerate(json_data["Time Series (1min)"][str(time_value)]):
		
			if str(value) != "5. volume":
				aux.append(float(json_data["Time Series (1min)"][str(time_value)][str(value)]))
			
		aux2.append(aux)

	input_data = np.array(aux2)
	input_data = input_data.transpose()

	print("Dimensions del input data")
	print(input_data.shape)

	# Save query response to json file
	with codecs.open('input_data.json', 'w', 'utf-8') as outfile:
		json_file = json.loads(response_intraday_data.text)
		data = json.dumps(json_file, indent=2, ensure_ascii=True)
		outfile.write(data)

	return input_data

# Function to get SMA data to be used as labels
def __get_labels():
	# Query to get SMA indicator on specific data (open, high, low & close)
	kind = ["open","high","low","close"]
	whole_SMA_respones = []
	for i,series_type in enumerate(kind):
		url = "https://www.alphavantage.co/query"
		querystring = {"apikey":"7CJAFX87TQG5584E","function":"SMA","symbol":"MSFT","interval":"1min","time_period":str(num_to_average),"series_type":series_type}
		headers = {
			'Cache-Control': "no-cache",
			'Postman-Token': "e7952a20-1d23-8df3-4d08-9f3579c0b095"
			}
		response_SMA = requests.request("GET", url, headers=headers, params=querystring)
		whole_SMA_respones.append(response_SMA)

	# Put labels in numpy array (n,m) where n=100 or ? (compact or full) and m=1 (number of indicators).
	aux2 = []
	for n,series_type in enumerate(whole_SMA_respones):
		aux = []
		
		SMA_json_data = json.loads(series_type.text)
		
		for i,time_value in enumerate(SMA_json_data["Technical Analysis: SMA"]):
			
			for j,value in enumerate(SMA_json_data["Technical Analysis: SMA"][str(time_value)]):
				aux.append(float(SMA_json_data["Technical Analysis: SMA"][str(time_value)][str(value)]))
				
		aux2.append(aux)	
		
	output_labels = np.array(aux2)
	#output_labels = output_labels.transpose()

	print("Dimensions del output data")
	print(output_labels.shape)

	# Save query response to json file
	with codecs.open('output_labels.json', 'w', 'utf-8') as outfile:
		json_file = json.loads(response_SMA.text)
		data = json.dumps(json_file, indent=2, ensure_ascii=True)
		outfile.write(data)
	
	return output_labels

# Preprocess the data to be fitted in a keras model
def preprocess_data(input_data,labels):
	
	data = []
	for n,intraday_values in enumerate(input_data):
		dummy = intraday_values.tolist()
		for i,value in enumerate(dummy):
			
			if (i+num_to_average)<=(len(intraday_values)):
				data.append(np.array(dummy[i:(i+num_to_average)]))
		
	real_data = np.array(data)
		
	labels = [[value] for i,value in enumerate(labels.flatten().tolist())]
	labels = np.array(labels)
	
	
	return real_data, labels

##### NN #####

# FUnction to train & test a model
def use_model(train_x,train_y,x):
	#Create a model
	model = Sequential()
	model.add(Dense(80, activation='relu', input_dim=num_to_average))
	model.add(Dense(40, activation='relu'))
	model.add(Dense(20, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	#Compile a model
	model.compile(loss='binary_crossentropy',
				  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
				  
	# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
	model.fit(train_x, train_y, epochs=5, batch_size=100)
	
	#generate predictions on new data
	classes = model.predict(x, batch_size=100)
	
	return classes

	
def main():

	#Get intraday tradking data and SMA indicators
	input_data = __get_input_data()
	labels = __get_labels()
	
	#Preprocess data to be ingested in NN
	processed_input_data, processed_labels = preprocess_data(input_data,labels)
	
	print(processed_input_data.shape)
	print(processed_labels.shape)
	
	#Create NN, train, and predict.
	prediction = use_model(processed_input_data, processed_labels,processed_input_data[1:100])
	
	
if __name__ == '__main__':
    main()
