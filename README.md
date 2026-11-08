# Hurricane Trajectory Prediction with an LSTM
In this project, we use an LSTM to predict the trajectories of hurricanes. We use the [Hurricane Database dataset from Unisys Weather](http://weather.unisys.com/hurricane/atlantic/). Here we preprocess the data, train a recurrent neural network on some of the samples, and compare our predictions against the real hurricanes. Uses data from 1851-2023.

WARNING: Some of the data may contain NaNs, which will throw an exception; to prevent this, search through hurricane_data.csv for them.

## Getting Started
All the data is in hurricane_data.csv. Preprocess the data using preprocess_data.py, then build and train the LSTM using train.py. Use predictor.py to visualize model predictions on a map. (Copy-paste the datapoints from a hurricane in all_data.csv into a new file (hurricane.csv).)

### Main Tools Necessary
- [Keras](https://keras.io/models/model/)
- [TensorFlow](tensorflow.org)
- [NumPy](numpy.org)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

## Modifications
1. Trains on and predicts raw latitude and longitude instead of combining these values into a grid ID. Combining them caused the model to only predict one reasonably; the other value would essentially be random. With both values used independently, predictions became much more reasonable
2. Uses all data from 1851-2023 instead of 2000-2009
3. Can visualize and plot model predictions on a map
4. Allows for the model to take its previous predictions as input, allowing for entire predicted trajectories rather than a singular "next point". However, accuracy decreases as more points are predicted.

## Limitations
While the model can predict reasonable-looking trajectories, they often vary wildly from the actual paths. The most likely reason is that external factors (such as the weather AROUND the hurricane rather than inside it) are more useful for predicting trajectories than internal factors. In the future, implementing a model trained on these systems would likely increase accuracy.
