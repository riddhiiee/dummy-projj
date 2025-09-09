#make env
#install lib
# software dev and ML dev
#software dev is static (update once a year)
#ml is continous deployment and updation
#pip freeze > requirements.txt gives all the versions of all the modules
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Dummy training data (XOR problem)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Define model
model = keras.Sequential([
    keras.layers.Dense(8, activation="relu", input_shape=(2,)),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X, y, epochs=100, verbose=0)

# Save model
model.save("my_model.h5")
print("✅ Model saved as my_model.h5")


#pip install reuirements.txt will install all the files