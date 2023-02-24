from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np

# Define the system parameters
N = 64  # number of reflecting elements
f_c = 28e9  # carrier frequency
lambda_c = 3e8 / f_c  # wavelength
d = lambda_c / 2  # element spacing
theta_inc = 30  # incident angle
P = 1  # transmit power
sigma = 1  # noise variance
alpha = 0.5  # reflection coefficient

# Generate the channel matrix
theta = np.linspace(-90, 90, N) * np.pi / 180
a = np.exp(1j * 2 * np.pi * d / lambda_c * np.outer(np.sin(theta), np.arange(N)))
h = alpha * a[:, np.argmax(theta == theta_inc)]  # channel for incident angle

# Define the neural network model
model = Sequential()
model.add(Dense(32, input_dim=N))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(N))
model.add(Activation('sigmoid'))

# Compile the model
optimizer = Adam(lr=0.01)
model.compile(loss='mse', optimizer=optimizer)

# Train the model on the channel matrix
X_train = np.zeros((1, N))
Y_train = np.abs(h) ** 2 / (P + sigma)
model.fit(X_train, Y_train, epochs=100)

# Use the model to design the IRS
X_test = np.zeros((1, N))
h_irs = alpha * a.dot(model.predict(X_test).T)  # reflected channel matrix


