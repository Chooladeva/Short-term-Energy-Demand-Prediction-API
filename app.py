from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import traceback
from waitress import serve 

app = Flask(__name__)

# Load the models from the pickle files
with open('nsw_model.pickle', 'rb') as f:
    nsw_model = pickle.load(f)

with open('sa_model.pickle', 'rb') as f:
    sa_model = pickle.load(f)

with open('qld_model.pickle', 'rb') as f:
    qld_model = pickle.load(f)

with open('vic_model.pickle', 'rb') as f:
    vic_model = pickle.load(f)

with open('tas_model.pickle', 'rb') as f:
    tas_model = pickle.load(f)

def predict_demand_for_region(Precipitation, RelativeHumidity, AirTemperature, WetBulbTemperature, DewTemperature,
                              SeaPressure, StationPressure, Month, Day, DayOfYear, Hour,
                              Season_Autumn, Season_Spring, Season_Summer, Season_Winter,
                              TimeOfDay_Afternoon, TimeOfDay_Evening, TimeOfDay_Morning, TimeOfDay_Night,
                              IsWeekend_False, IsWeekend_True, model):
    predicted_demand_next_24_hours = []
    for hour in range(24):
        # Prepare the input features for prediction
        input_features = [[Precipitation, RelativeHumidity, AirTemperature, WetBulbTemperature, DewTemperature,
                           SeaPressure, StationPressure, Month, Day, Hour + hour, DayOfYear,
                           Season_Autumn, Season_Spring, Season_Summer, Season_Winter,
                           TimeOfDay_Afternoon, TimeOfDay_Evening, TimeOfDay_Morning, TimeOfDay_Night,
                           IsWeekend_False, IsWeekend_True]]
        
        # Make predictions using the provided model
        predicted_demand = model.predict(input_features)[0]
        
        predicted_demand_next_24_hours.append(predicted_demand)
    
    # Create a list of hour values from 1 to 24
    hours = list(range(1, 25))
    
    # Create a DataFrame with 'Hour' as the index and 'Predicted Demand' as the column
    predicted_demand_df = pd.DataFrame({'1_Hour': hours, '2_Predicted Demand (MW)': predicted_demand_next_24_hours})
    # Calculate cumulative sum of predicted demands
    predicted_demand_df['3_Cumulative Demand (MW)'] = predicted_demand_df['2_Predicted Demand (MW)'].cumsum()
    # Calculate average hourly predicted demand
    predicted_demand_df['4_Average Demand per hour (MW)'] = predicted_demand_df['3_Cumulative Demand (MW)'] / (predicted_demand_df['1_Hour'])
    
    # Reorder the columns as specified
    #ordered_columns = ['1_Hour', 'Predicted Demand (MW)', 'Cumulative Demand (MW)', 'Average Demand per hour (MW)']
    #predicted_demand_df = predicted_demand_df[ordered_columns]

    return predicted_demand_df
    

# Route for rendering the index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and display predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        region = request.form['region']  # Get the selected region from the form
        # Extract input data from the form
        input_data = {
            'Precipitation': float(request.form['precipitation']),
            'RelativeHumidity': float(request.form['relative_humidity']),
            'AirTemperature': float(request.form['air_temperature']),
            'WetBulbTemperature': float(request.form['wet_bulb_temperature']),
            'DewTemperature': float(request.form['dew_point_temperature']),
            'SeaPressure': float(request.form['sea_pressure']),
            'StationPressure': float(request.form['station_pressure']),
            'Month': int(request.form['month']),
            'Day': int(request.form['day_of_month']),
            'DayOfYear': int(request.form['day_of_year']),
            'Hour': int(request.form['hour']),
            'Season_Autumn': 1 if request.form['season'] == 'Autumn' else 0,
            'Season_Spring': 1 if request.form['season'] == 'Spring' else 0,
            'Season_Summer': 1 if request.form['season'] == 'Summer' else 0,
            'Season_Winter': 1 if request.form['season'] == 'Winter' else 0,
            'TimeOfDay_Afternoon': 1 if request.form['time_of_day'] == 'Afternoon' else 0,
            'TimeOfDay_Evening': 1 if request.form['time_of_day'] == 'Evening' else 0,
            'TimeOfDay_Morning': 1 if request.form['time_of_day'] == 'Morning' else 0,
            'TimeOfDay_Night': 1 if request.form['time_of_day'] == 'Night' else 0,
            'IsWeekend_False': 1 if request.form['weekday_weekend'] == 'weekday' else 0,
            'IsWeekend_True': 1 if request.form['weekday_weekend'] == 'weekend' else 0
        }

        # Select the appropriate model based on the selected region
        if region == 'New South Wales':
            model = nsw_model
        elif region == 'South Australia':
            model = sa_model
        elif region == 'Queensland':
            model = qld_model
        elif region == 'Victoria':
            model = vic_model
        elif region == 'Tasmania':
            model = tas_model
        else:
            return "Invalid region"

        # Make predictions using the selected model and input data
        predicted_demand_df = predict_demand_for_region(**input_data, model=model)
        
        # Round the values in the dataframe to 3 decimal places
        predicted_demand_df_rounded = predicted_demand_df.round(3)
        
        # Convert the dataframe to a dictionary with ordered columns
        predicted_demand_dict = predicted_demand_df_rounded.to_dict(orient='records')
        
        # Return the dictionary in the API response
        return jsonify(predicted_demand_dict)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8000)