import tensorflow as tf
import numpy as np


def train_model(input_xs, output_ys):
    # cria um modelo sequencial
    model = tf.keras.Sequential()

    # adiciona camadas ao modelo
    model.add(tf.keras.layers.Dense(units=80, activation="relu", input_shape=(7,)))
    model.add(tf.keras.layers.Dense(units=3, activation="softmax"))

    # compila o modelo
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # treina o modelo
    model.fit(input_xs, output_ys, epochs=100, shuffle=True, verbose=0)

    return model


# Dados de entrada
tensor_pessoas = [
    [0.33, 1, 0, 0, 1, 0, 0],  # Erick
    [0, 0, 1, 0, 0, 1, 0],  # Ana
    [1, 0, 0, 1, 0, 0, 1],  # Carlos
]

# Labels (one-hot)
tensor_labels = [
    [1, 0, 0],  # premium
    [0, 1, 0],  # medium
    [0, 0, 1],  # basic
]

# converte os dados para tensores do TensorFlow
input_xs = tf.constant(tensor_pessoas, dtype=tf.float32)
output_ys = tf.constant(tensor_labels, dtype=tf.float32)

# treina o modelo
model = train_model(input_xs, output_ys)

# criando um novo tensor para a pessoa jose
pessoa = {"nome": "jose", "idade": 28, "cor": "verde", "localizacao": "São Paulo"}

# pessoa normalizada
jose = [
    [
        0.28,
        0,
        0,
        1,
        1,
        0,
        0,
    ],
]

jose_input = tf.constant(jose, dtype=tf.float32)
pred: np.ndarray = model.predict(jose_input, verbose=0)

classes = ["premium", "medium", "basic"]

probs = pred[0]
for nome, p in zip(classes, probs):
    print(f"{nome}: {p * 100:.2f}%")
