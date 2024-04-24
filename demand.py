import pickle
import pandas as pd
import numpy as np

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
pd.options.display.max_columns = None

from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor


# Load resampled_data from the pickle file
def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Load data from the pickle file
data = load_from_pickle('data.pkl')

# Load the trained models and their best parameters from the pickle file
with open('trained_models.pickle', 'rb') as f:
    models, best_params = pickle.load(f)

# Create a dictionary to map regions to their corresponding models
regions = ['NSW', 'QLD', 'SA', 'TAS', 'VIC']
trained_models_dict = dict(zip(regions, models))

nsw_model = trained_models_dict['NSW']
nsw_best_params = best_params[regions.index('NSW')]

qld_model = trained_models_dict['QLD']
qld_best_params = best_params[regions.index('QLD')]

sa_model = trained_models_dict['SA']
sa_best_params = best_params[regions.index('SA')]

tas_model = trained_models_dict['TAS']
tas_best_params = best_params[regions.index('TAS')]

vic_model = trained_models_dict['VIC']
vic_best_params = best_params[regions.index('VIC')]


 # Define input and output columns
input_columns = ['Precipitation', 'RelativeHumidity%', 'AirTemperature','WetBulbTemperature', 'DewTemperature', 'SeaPressure', 
                 'StationPressure', 'Month', 'Day', 'Hour', 'DayOfYear', 'Season_Autumn', 'Season_Spring', 'Season_Summer', 
                 'Season_Winter', 'TimeOfDay_Afternoon', 'TimeOfDay_Evening','TimeOfDay_Morning', 'TimeOfDay_Night', 
                 'IsWeekend_False', 'IsWeekend_True'
                ]
output_columns = ["TotalDemand"]

# Function to create a LightGBM model with best parameters
def create_lgb_model(best_params):
    model = LGBMRegressor(**best_params, n_jobs=-1, random_state=42, verbose=-1)
    
    return model

# Function to train LightGBM model for a specific region
def train_lgb_model(data, region, input_columns, output_columns, best_params):
    
    # Filter data for the specified region
    region_data = data[data['Region'] == region]
    
    # Prepare the input features and target variable
    data_X = region_data[input_columns]
    data_y = region_data[output_columns]
    
    # Train-test split without shuffling
    random_state = np.random.RandomState(seed=42)
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, shuffle=False, random_state=random_state)
    
    # Create and fit the LightGBM model with best parameters
    model = create_lgb_model(best_params)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test


# Example usage for Region NSW
nsw_model, nsw_X_test, nsw_y_test = train_lgb_model(data, 'NSW', input_columns, output_columns, nsw_best_params)

# Example usage for Region SA
sa_model, sa_X_test, sa_y_test = train_lgb_model(data, 'SA', input_columns, output_columns, sa_best_params)

# Example usage for Region QLD
qld_model, qld_X_test, qld_y_test = train_lgb_model(data, 'QLD', input_columns, output_columns, qld_best_params)

# Example usage for Region QLD
vic_model, vic_X_test, vic_y_test = train_lgb_model(data, 'VIC', input_columns, output_columns, vic_best_params)

# Example usage for Region QLD
tas_model, tas_X_test, tas_y_test = train_lgb_model(data, 'TAS', input_columns, output_columns, tas_best_params)

# Save the model for NSW region to a pickle file
with open('nsw_model.pickle', 'wb') as f:
    pickle.dump(nsw_model, f)

# Save the model for SA region to a pickle file
with open('sa_model.pickle', 'wb') as f:
    pickle.dump(sa_model, f)
    
# Save the model for QLD region to a pickle file
with open('qld_model.pickle', 'wb') as f:
    pickle.dump(qld_model, f)

# Save the model for VIC region to a pickle file
with open('vic_model.pickle', 'wb') as f:
    pickle.dump(vic_model, f)
    
# Save the model for TAS region to a pickle file
with open('nsw_model.pickle', 'wb') as f:
    pickle.dump(tas_model, f)