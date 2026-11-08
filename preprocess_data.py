import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

data = pd.read_csv('hurricane_data.csv', names=['Year', 'Month', 'Day', 'Hour', 'HurricaneNum', 'Name', 'Lat', 'Long', 'WindSpeed', 'Pressure', 'Nullcol'])

# Create a unique key for all hurricanes
data['unique-key'] = data['Name'] + '-' + data['Year'].map(str) + '-' + data['HurricaneNum'].map(str)

# Delete the columns of information that we are not using so far
data.drop(['Name', 'HurricaneNum', 'Year', 'Pressure', 'Nullcol'], axis = 1, inplace = True)

# Since our keys are strings, we enumerate them to access them as integers
keys = list(enumerate(pd.unique(data['unique-key'])))

y = np.zeros((1260))
for x in range(1260):
    y[x] = len(pd.DataFrame(data[data['unique-key'] == keys[x][1]], columns = data.keys()).reset_index(drop = True))

# Now contains how many time instances of data (or rows) each hurricane contains
hurricane_amount = pd.DataFrame(y)

from geopy.distance import great_circle as vc
import math as Math

y = np.zeros((1260))
data['distance'] = np.zeros(36669)
data['direction'] = np.zeros(36669)

# For all hurricanes
for x in range(1260):
    t = pd.DataFrame(data[data['unique-key'] == keys[x][1]], columns=data.keys()).reset_index(drop=False)
    dst = 0
    prev = (0, 0)

    # For all latitude and longitude points of hurricane, calculate the angle of travel and distance
    for p in zip(t['Lat'], t['Long']):
        if prev == (0, 0):
            prev = p
            continue
            # Stores the distance into the DataFrame
        data._set_value(t[(t['Lat'] == p[0]) & (t['Long'] == p[1])]['index'].values[0], 'distance', vc(prev, p).miles)

        dLon = p[1] - prev[1];
        y_x = Math.sin(dLon) * Math.cos(p[0]);
        x_x = Math.cos(p[1]) * Math.sin(p[0]) - Math.sin(p[1]) * Math.cos(p[0]) * Math.cos(dLon);
        brng = Math.degrees(Math.atan2(p[1]-prev[1], p[0]-prev[0]))
        if brng < 0:
            brng += 360;

        # Stores the angle of travel into the DataFrame
        data._set_value(t[(t['Lat'] == p[0]) & (t['Long'] == p[1])]['index'].values[0], 'direction', brng)
        dst += vc(prev, p).miles
        prev = p
    y[x] = dst

# We are removing some outliers that contain too little or too much information to keep a more normal distribution.
cond = (hurricane_amount > 13) & (hurricane_amount < 60)
keys25 = []

for x in cond.index:
    if cond.loc[x][0]:
        keys25.append(keys[x][1])

word2keys = {}
for x in keys:
    word2keys[x[1]] = x[0]

df = data[data['unique-key'].isin(keys25)]

# Assigning each point to a specific location in the grid.
df['gridID'] = np.zeros(26965)

df['gridID'] = (df['Lat'] - 9.500) + ((df['Long'] + 107.700))*240
df['gridID'] = df['gridID'] + 7040
df['gridID'] = round(df['gridID'])

df.head()

df.to_csv('all_data.csv') # Save the dataframe to csv for checkpoint
