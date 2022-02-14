# TODO:
# - Implement custom activation functions to CNN
#   (to measure accuracy with Chebyshev's polynomial approximation)
# - Find optimal structure for CNN model
# - Save/read model
# - Utilize Google Colab for training (?)
# - Rearrange&improve code

from keras.datasets import mnist
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K


# Data preprosessing
(train_X, train_y), (test_X, test_y) = mnist.load_data()
x_train = train_X.astype("float32") / 255
x_test = test_X.astype("float32") / 255
num_classes = 10
input_shape = (28, 28, 1)
y_train = keras.utils.to_categorical(train_y, num_classes)
y_test = keras.utils.to_categorical(test_y, num_classes)

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

# Create model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(5, kernel_size=(1, 1), activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Conv2D(10, kernel_size=(1, 1), activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128),
        layers.Activation('relu'),
        layers.Dropout(0.75),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 120
epochs = 50
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])



