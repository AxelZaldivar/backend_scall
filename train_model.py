import numpy as np
import tensorflow as tf
from tensorflow import keras

# Datos de ejemplo (XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Definir modelo
model = keras.Sequential([
    keras.layers.Dense(4, activation="relu", input_shape=(2,)),
    keras.layers.Dense(1, activation="sigmoid")
])

# Compilar y entrenar
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=100, verbose=0)

# Guardar modelo
model.save("model.h5")
print("Modelo guardado correctamente.")
