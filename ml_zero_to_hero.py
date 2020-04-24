import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_imgs, train_labs), (test_imgs, test_labs) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_imgs[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labs[i]])
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

model.fit(train_imgs, train_labs, epochs=30)

# probability_model = tf.keras.Sequential([model.tf.keras.layers.Softmax()])

# predictions = probability_model.predict(test_imgs)

# ans = np.argmax(predictions[0])

# print(class_names[ans])
