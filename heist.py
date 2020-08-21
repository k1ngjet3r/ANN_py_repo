import numpy as np
import tensorflow as tf

<<<<<<< HEAD
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
=======
# mnist = tf.keras.datasets.mnist
>>>>>>> 5c746c8ed3c32617939d2285c7ac50eef586ef3e

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test/255.0
# print(x_train[0])
# print((x_train[0]).shape)

<<<<<<< HEAD
model = keras.Sequential(
    [
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_images, train_labels, epochs=30)
=======

a = np.zeros((3, 3), 'd')
print(a)

b = np.linspace(0, 10, 5)
>>>>>>> 5c746c8ed3c32617939d2285c7ac50eef586ef3e

print(b)

c = np.array((2, 4))
print(c)