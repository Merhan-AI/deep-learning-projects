import tensorflow as tf
from tensorflow import keras
import numpy as np
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(1,)),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=1000, verbose=0)

loss, accuracy= model.evaluate(X,y)
print("Accuracy:",accuracy)

print(model.predict([3]))
print(model.predict([8]))