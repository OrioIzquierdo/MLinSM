import numpy as np
import requests
import json
import codecs
import math
import random
import matplotlib.pyplot as plt

##### VARIABLES #####
num_simulations = 1000
time_simulations = 120

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

# Preprocess the data to be fitted in a keras model
def preprocess_data(input_data):
	
    #Make Montecarlo on open intraday data.
	return input_data[0][:time_simulations]

##### MONTECRALO #####

def monte_carlo_sim(prices, num_simulations):
    
    last_price = prices[-1]
    simulated_prices = []
        
    for j in range(num_simulations):
    
        changes = []
        random_numbers = []
        
        #Compute and store the changes in every minute & create a list of random numbers.
        for i in range(len(prices) - 1):
        
            changes.append(math.log(prices[i]/prices[i+1]))
            random_numbers.append(random.randint(0,len(prices)-2))
        
        ordered_changes = sorted(changes)
        random_changes = []
        generated_prices = []
        
        #Get random change and generate simulated prices.
        for i in range(len(prices) - 1):
            #print(i)
            if i==0:
                random_changes.append(ordered_changes[random_numbers[i]])
                generated_prices.append(last_price*math.exp(random_changes[i]))
                
            else:      
                random_changes.append(ordered_changes[random_numbers[i]])
                generated_prices.append(generated_prices[i-1]*math.exp(random_changes[i]))
                
        simulated_prices.append(generated_prices)
    
    return simulated_prices
 
##### PLOT SIMULATION #####

def plot_simulation(y):

	for j,y_data in enumerate(y):
		x = []
		for i in range(len(y_data)):
			x.append(time_simulations*i/len(y_data))
                 
		plt.plot(x, y_data, '-o', markersize=2)
        
    
    # Set the axis from -1 to 1, set x axis to invisble, and create 2 lines crossing the plot
	flat_list = [item for sublist in y for item in sublist]
	plt.axis([min(x), max(x), min(flat_list), max(flat_list)])
	plt.ylabel("price")
	plt.xlabel("time (min)")
    
    # Save the plot into a png file
	plt.savefig("Montecarlo_simulation.png")

    # Clear the plot to draw the next intent
	plt.clf()

 
def main():

	#Get intraday tradking data and SMA indicators
    input_data = __get_input_data()
	
	#Preprocess data to be ingested in NN
    processed_input_data = preprocess_data(input_data)
	
    print(processed_input_data.shape)
    	
    simulated_prices = monte_carlo_sim(processed_input_data, num_simulations)
	
    print(len(simulated_prices))
    print(len(simulated_prices[0]))
    
    plot_simulation(simulated_prices)
	
	
if __name__ == '__main__':
    main()
