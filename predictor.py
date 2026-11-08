import numpy as np
import pandas as pd
import tensorflow as tf
from math import atan2, degrees
from geopy.distance import great_circle as vc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

df0 = pd.read_csv('hurricane.csv', index_col=0)
lat = df0['Lat'].values
long = df0['Long'].values
df0.drop(['Month', 'Day', 'Hour', 'unique-key'], axis = 1, inplace = True)
model3 = tf.keras.models.load_model('hurricane3.keras')
df0['WindSpeed'] = (df0['WindSpeed'].values / 160)
df0['distance'] = (df0['distance'].values / 3561)
df0['direction'] = (df0['direction'].values / 360)
df0['Lat'] = df0['Lat'].values/81
df0['Long'] = (df0['Long'].values+137)/200
data = df0.values
df3 = []
df3.append(data[0:14])

for i in range(len(data)-15):
    df4 = df3[i-1]
    next_row = []
    thing2 = []
    thing2.append(df3[i-1])
    thing = np.array(thing2)
    df2 = thing[:, :]
    df2 = np.reshape(df2, (df2.shape[0], df2.shape[1], 5))
    df2 = np.asarray(df2).astype('float32')
    pred = model3.predict(df2)
    good_lat = pred[0][0]*81
    good_long = pred[0][1]*200-127
    prev = df3[i-1][13][3]
    prev_lat = lat[12+i]
    prev_long = long[12+i]
    next_row.append(good_lat/81)
    next_row.append((good_long+127)/200)
    next_row.append(float(df3[0][13][0]))
    next_row.append(vc((good_lat, good_long), (prev_lat, prev_long)).miles/3561)
    direction = degrees(atan2(good_lat-prev_lat, good_long-prev_long))
    if direction < 0:
        direction+=360
    next_row.append(direction/360)
    df4 = df4.tolist()
    first_row = df4.pop(0)
    df4.append(next_row)
    df4 = np.array(df4, dtype=object)
    df3.append(df4)

thing = np.array(df3)
df = thing[:, :-1]
df = np.reshape(df, (df.shape[0], df.shape[1], 5))
df = np.asarray(df).astype('float32')
pred = model3.predict(df)
latlist = []
longlist = []
coordinates = []
for i in pred:
    latlist.append(i[0]*81)
    longlist.append(i[1]*200-127)
thing = latlist.pop(0)
latlist.append(thing)
thing = longlist.pop(0)
longlist.append(thing)

# Correction Code
n1 = lat[13]-latlist[0]
n2 = long[13]-longlist[0]
for i in range(len(latlist)):
    latlist[i]+=n1
for i in range(len(longlist)):
    longlist[i]+=n2
for i in range(len(latlist)):
    coordinates.append((latlist[i-1], longlist[i-1]))

f = open('predicted_trajectory.txt', 'w')
for i in coordinates:
    i = str(i)
    i = i.replace('(', '')
    i = i.replace(')', '')
    f.write(f'{i}\n')
f.close()

f = open('real_trajectory.txt', 'w')
for i in range(len(lat)):
    f.write(f'{str(lat[i])}, {str(long[i])}\n')
f.close()
latlist1 = []
longlist1 = []
latlist2 = []
longlist2 = []
f1 = open('predicted_trajectory.txt', 'r')
f2 = open('real_trajectory.txt', 'r')
for line in f1:
    line = line.replace(',', '')
    lat, long = line.split()
    latlist1.append(float(lat))
    longlist1.append(float(long))
for line in f2:
    line = line.replace(',', '')
    lat, long = line.split()
    latlist2.append(float(lat))
    longlist2.append(float(long))
latlist4 = latlist2[-(len(latlist2)-13):-1]
longlist4 = longlist2[-(len(latlist2)-13):-1]
differencelist = []
for i in range(len(latlist1)):
    differencelist.append(vc((latlist1[i-1], longlist1[i-1]), (latlist4[i-1], longlist4[i-1])).miles)
average = 0
for i in differencelist:
    average+=i
print(average/len(differencelist))
latlist3 = []
longlist3 = []
for i in range(len(latlist1)):
    latlist3.append(abs(latlist1[i-1]-latlist4[i-1]))
    longlist3.append(abs(longlist1[i-1]-longlist4[i-1]))
average1 = 0
average2 = 0
for i in range(len(latlist3)):
    average1+=latlist3[i-1]
    average2+=longlist3[i-1]
print(average1/len(latlist3))
print(average2/len(longlist3))

average1 = 0
average2 = 0
for i in range(len(latlist3)):
    average1+=latlist3[i-1]
    average2+=longlist3[i-1]
print(average1/len(latlist3))
print(average2/len(longlist3))
# Create a Basemap instance
plt.figure(figsize=(10, 6))
m = Basemap(projection='ortho', lon_0=-75, lat_0=20)
m.drawcoastlines()
m.drawcountries()
predicted_x, predicted_y = m(longlist1, latlist1)
actual_x, actual_y = m(longlist2, latlist2)
m.plot(actual_x, actual_y, marker='x', color='blue', label=f'Actual Path')
m.plot(predicted_x, predicted_y, marker='o', linestyle='dashed', color='orange', label=f'Predicted Path')
plt.title(f'Actual vs Predicted Path')
plt.legend(loc='best')

plt.show()
