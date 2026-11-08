import numpy as np
import pandas as pd
import tensorflow as tf
from math import atan2, degrees
from geopy.distance import great_circle as vc
df = pd.read_csv('hurricane.csv', index_col=0)
table = open('table.txt', 'r')
lines = []
for line in table:
    line = line.replace(',', '')
    lines.append(line.split())
lat = df['Lat'].values
long = df['Long'].values
df.drop(['Month', 'Day', 'Hour', 'unique-key'], axis = 1, inplace = True)
model = tf.keras.models.load_model('hurricane.keras')
df['gridID'] = (df['gridID'].values / 41040)
df['WindSpeed'] = (df['WindSpeed'].values / 160)
df['distance'] = (df['distance'].values / 3561)
df['direction'] = (df['direction'].values / 360)
df['Lat'] = df['Lat'].values/81
df['Long'] = (df['Long'].values+137)/200
data = df.values
df3 = []
df3.append(data[0:14])

for i in range(len(data)-15):
    df4 = df3[i-1]
    next_row = []
    thing2 = []
    thing2.append(df3[i-1])
    thing = np.array(thing2)
    df2 = thing[:, :]
    df2 = np.reshape(df2, (df2.shape[0], df2.shape[1], 6))
    df2 = np.asarray(df2).astype('float32')
    pred = model.predict(df2)
    pred2 = round(pred[0][0]*41040)+7040
    prev = df3[i-1][13][3]
    latlong = []
    for j in lines:
        if pred2 == int(j[2]):
            latlong.append(j)
    prev_lat = lat[12+i]
    prev_long = long[12+i]
    for j in latlong:
        j.append(vc((float(j[0]), float(j[1])), (prev_lat, prev_long)).miles)
    good_lat = float(latlong[-1][0])
    good_long = float(latlong[-1][1])
    np.append(lat, good_lat)
    np.append(long, good_long)
    next_row.append(good_lat/81)
    next_row.append((good_long+137)/200)
    next_row.append(float(df3[0][13][0]))
    next_row.append(latlong[-1][3]/3561)
    direction = degrees(atan2(good_lat-prev_lat, good_long-prev_long))
    if direction < 0:
        direction+=360
    next_row.append(direction/360)
    next_row.append(pred[0][0])
    print(next_row)
    df4 = df4.tolist()
    first_row = df4.pop(0)
    df4.append(next_row)
    df4 = np.array(df4, dtype=object)
    df3.append(df4)

thing = np.array(df3)
df = thing[:, :-1]
df = np.reshape(df, (df.shape[0], df.shape[1], 6))
df = np.asarray(df).astype('float32')
pred = model.predict(df)
gridlist = []
coordinates = []
for i in pred:
    gridlist.append(round(i[0]*41040)+7040)
for i in gridlist:
    latlong = []
    for j in lines:
        if int(i) == int(j[2]):
            latlong.append(j)
    good_lat = float(latlong[-1][0])
    good_long = float(latlong[-1][1])
    coordinates.append((good_lat, good_long))
differencelist = []
for i in range(len(coordinates)):
    differencelist.append(vc(coordinates[i-1], (lat[14+i], long[14+i])).miles)
f = open('real_trajectory.txt', 'w')
for i in range(55):
    f.write(f'{str(lat[i-1])}, {str(long[i-1])}\n')
f.close()
f = open('predicted_trajectory.txt', 'w')
for i in coordinates:
    i = str(i)
    i = i.replace('(', '')
    i = i.replace(')', '')
    f.write(f'{i}\n')
f.close()
average = 0
for i in differencelist:
    average+=i
print(average/len(differencelist))
table.close()
longlist = []
for i in range(len(coordinates)):
    longlist.append(abs(coordinates[i][1]-long[i]))
average = 0
for i in longlist:
    average+=i
print(average/len(longlist))
latlist = []
for i in range(len(coordinates)):
    latlist.append(abs(coordinates[i][1]-lat[i]))
average = 0
for i in latlist:
    average+=i
print(average/len(latlist))
