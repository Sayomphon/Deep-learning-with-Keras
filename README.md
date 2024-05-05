# Deep-learning-with-Keras
A. Build a baseline model (5 marks) 
Use the Keras library to build a neural network with the following:
- One hidden layer of 10 nodes, and a ReLU activation function
- Use the adam optimizer and the mean squared error  as the loss function.
  1. Randomly split the data into a training and test sets by holding 30% of the data for testing. You can use the 
train_test_split
helper function from Scikit-learn.
  2. Train the model on the training data using 50 epochs.
  3. Evaluate the model on the test data and compute the mean squared error between the predicted concrete strength and the actual concrete strength. You can use the mean_squared_error function from Scikit-learn.
  4. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.
  5. Report the mean and the standard deviation of the mean squared errors.

# Code
#To build the baseline model, we will first import the required libraries and load the dataset into a Pandas dataframe.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

# Load dataset
concrete_data = pd.read_csv('concrete_data.csv')

#Next, we will split the dataset into a training and test set using the train_test_split function from Scikit-learn.

# Split dataset into features X and target variable y
X = concrete_data.iloc[:, :-1]
y = concrete_data.iloc[:, -1]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#We will now build the neural network model with one hidden layer of 10 nodes, using the ReLU activation function, and compiling it with the adam optimizer and mean squared error loss function.

# Define model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

#We will train the model on the training data with 50 epochs.

# Train model
model.fit(X_train, y_train, epochs=50, verbose=0)

#Next, we will evaluate the model on the test data and compute the mean squared error.

# Evaluate model on test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(mse)

#To repeat the above steps 50 times and compute the mean and standard deviation of the mean squared errors, we will create a list of 50 mean squared errors and then calculate the mean and standard deviation.

# Repeat steps 1-3, 50 times
mse_list = []
for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    model.fit(X_train, y_train, epochs=50, verbose=0)
    y_pred = model.predict(X_test)
    mse_list.append(mean_squared_error(y_test, y_pred))

# Calculate mean and standard deviation of mean squared errors
mean_mse = np.mean(mse_list)
std_mse = np.std(mse_list)
print("Mean Squared Error:", mean_mse)
print("Standard Deviation:", std_mse)

#The final output will report the mean and standard deviation of the 50 mean squared errors computed in the above step.

