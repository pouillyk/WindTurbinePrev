import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from forecast import hourly_dataframe as hd

#Convert wind speed in m/s
hd['wind_speed_10m']=hd['wind_speed_10m']*1000/3600
#Load Model
model = joblib.load('bestModelTurbine.pkl')

#Function to predict power output
def predict_power(windSpeed, WindDir):
    u = windSpeed * np.cos(np.radians(WindDir))
    v = windSpeed * np.sin(np.radians(WindDir))
    features = np.array([[windSpeed, WindDir, u, v, windSpeed**2, windSpeed**3]])
    
    return model.predict(features)[0]

#Variable to sum all power over the day
totPow=0

for i in range(len(hd)):
    power_pred = predict_power(windSpeed=hd['wind_speed_10m'][i], WindDir=hd['wind_direction_10m'][i])
    #Forcast is given hourly when model is sampled at 10 min
    power_pred*=6
    print(f"Predicted Power Output between {i}:00 and {i+1}:00 : {power_pred:.2f} kW")
    totPow+=power_pred

print(f"Total predicted Power Output for the next day  : {totPow:.2f} kW")
print(f"Mean wind speed at 10 meters: {np.mean(hd['wind_speed_10m']):.2f} m/s")
