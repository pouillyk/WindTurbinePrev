import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt


print('### Loading Dataset ###')
#Load dataset as a DataFrame
df_in = pd.read_csv('data/Aventa_AV7_IET_OST_SCADA.csv')


print('### Data cleaning ###')

df_in=df_in.dropna()

df_in["Datetime"] = pd.to_datetime(df_in["Datetime"])
df_in = df_in.set_index("Datetime")

df = df_in.copy()



lenIData = len(df)
print('Initial data length', lenIData)

# Data Cleaning
# Turbine characteristics
# https://en.wind-turbine-models.com/turbines/1529-aventa-av-7


#Cut-in / Cut-Out
df[df.WindSpeed<2]=0.0
df[df.WindSpeed>14]=0.0

# Wind turbine stopped
df = df.drop(df[df.PowerOutput < 0].index)

#Curtailment
df = df.drop(df[(df.PowerOutput<6.2) & (df.WindSpeed>6.5)].index)

#Turbine stopped 
df.drop(df[(df.RotorSpeed==0)&(df.WindSpeed>3)].index)
#Turbine over-heated:
df.drop(df[df.GeneratorTemperature>1000].index)

#Feature Engineering

#Conversion the wind speed and direction to North-South and East-West wind components
#The angle is given with respect to the turbine direction (North-East), we thus substracted 45 degrees to get de wind direction
df['WindDirection'] = df['offsetWindDirection']-45
df['Wind_N-S']=df['WindSpeed']*np.cos(np.radians(df['WindDirection'])) 
df['Wind_E-W']=df['WindSpeed']*np.sin(np.radians(df['WindDirection'])) 

#Resampling at 10 min
df_10m = pd.DataFrame()

# Wind : Mean, Standard Deviation and Max (Turbulence)
df_10m["WindSpeed"] = df["WindSpeed"].resample("10T").mean()
df_10m["WindSpeedStd"] = (
    df["WindSpeed"].resample("10T").std()
)
df_10m["WindSpeedMax"] = df["WindSpeed"].resample("10T").max()

# Direction : Vectorial mean
u_mean = df["Wind_N-S"].resample("10T").mean()
v_mean = df["Wind_E-W"].resample("10T").mean()
df_10m["WindDirection"] = (np.degrees(np.arctan2(v_mean, u_mean))) % 360
df_10m["Wind_N-S"] = u_mean
df_10m["Wind_E-W"] = v_mean

# Power and mechanic variables: mean
df_10m["PowerOutput"] = df["PowerOutput"].resample("10T").mean()
df_10m["RotorSpeed"] = df["RotorSpeed"].resample("10T").mean()
df_10m["GeneratorSpeed"] = df["GeneratorSpeed"].resample("10T").mean()
df_10m["GeneratorTemperature"] = df["GeneratorTemperature"].resample("10T").mean()

#The power output theoritically depend on the wind speed to the power 3
df_10m['Windp2'] = df_10m['WindSpeed']**2
df_10m['Windp3'] = df_10m['WindSpeed']**3


#Clean highly turbulent wind
df_10m.drop(df_10m[df_10m.WindSpeedStd/df_10m.WindSpeed>0.3].index)

# Clean NaN
df_10m=df_10m.dropna()
lenFData = len(df_10m)
print('Final data length', lenFData)

#Save cleaned dataset
df_10m.to_csv('data/dataTurbine_clean.csv')

# Plot saved Data Power vs Wind Speed
print('Plotting...')

plt.figure()
plt.scatter(df_10m['WindSpeed'], df_10m['PowerOutput'])
plt.xlabel('Wind Speed')
plt.ylabel('Power Output')

plt.show()