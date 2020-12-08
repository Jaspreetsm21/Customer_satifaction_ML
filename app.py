from flask import Flask, request, render_template
#from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np 
# Load the Random Forest CLassifier model
file_name = "model_file_1.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()    
    if request.method == 'POST':
        Age = str(request.form['Age'])
        Flight_Distance = int(request.form['Flight_Distance'])
        Seat_comfort=int(request.form['Seat_comfort'])
        time_convenient=int(request.form.get('time_convenient',0))		
        Food_and_drink=int(request.form['Food_and_drink'])
        Gate_location=int(request.form['Gate_location'])
        Inflight_wifi_service =int(request.form['Inflight_wifi_service'])
        Inflight_entertainment =int(request.form['Inflight_entertainment'])
        Online_support =int(request.form['Online_support'])
        Ease_of_Online_booking =int(request.form['Ease_of_Online_booking'])
        On_board_service =int(request.form.get('On_board_service',0))
        Leg_room_service =int(request.form['Leg_room_service'])
        Baggage_handling =int(request.form['Baggage_handling'])
        Checkin_service =int(request.form['Checkin_service'])
        Cleanliness =int(request.form['Cleanliness'])
        Online_boarding =int(request.form['Online_boarding'])
        Departure_Delay_in_Minutes =int(request.form['Departure_Delay_in_Minutes']	)	
        Arrival_Delay_in_Minutes =int(request.form['Arrival_Delay_in_Minutes'])
        Gender_Female =request.form.get('Gender_Female',0)
        Gender_Male =request.form['Gender_Male']
        Type_Loyal_customer =request.form['Type_Loyal_customer']
        Type_disloyal_Customer =request.form['Type_disloyal_Customer']
        Travel_Business_travel =request.form['Travel_Business_travel']
        Travel_Personal_Travel =request.form['Travel_Personal_Travel']
        Class_Business =request.form['Class_Business']
        Class_Eco =request.form['Class_Eco']
        Class_Eco_Plus =request.form['Class_Eco_Plus']
        temp_array = [Age,Flight_Distance,Seat_comfort,	time_convenient,Food_and_drink,	Gate_location,	Inflight_wifi_service ,	Inflight_entertainment ,Online_support ,Ease_of_Online_booking ,	On_board_service ,	Leg_room_service ,	Baggage_handling ,	Checkin_service  ,	Cleanliness ,	Online_boarding ,	Departure_Delay_in_Minutes,Arrival_Delay_in_Minutes      ,   Gender_Female ,	 Gender_Male,	Type_Loyal_customer ,	Type_disloyal_Customer ,		Travel_Business_travel ,Travel_Personal_Travel ,Class_Business ,Class_Eco ,	Class_Eco_Plus ]
        data = np.array([temp_array])
        print(data)
        my_prediction = int((model.predict(data).reshape(1,-1))[0])
        my_prediction = np.where(my_prediction==1,'satisfied', 'unsatisfied')
        print(my_prediction)
        return render_template("result.html", my_prediction=my_prediction,data = data)
if __name__ == "__main__":
    app.run(debug=True)