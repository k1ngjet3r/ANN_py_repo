import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load and Preprocess the data
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

# Define Keras Model
model = tf.keras.Sequential([
    # transforms the format of the images from a 2D to 1D array
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Hidden layers, contain 128 nodes (neurons)
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layers, contain 10 outputs
    tf.keras.layers.Dense(10)
])

# Compile Keras Model
model.compile(
    # Optimizer: how the model is updated based on the data it sees and its loss function
    optimizer='sgd',
    # Loss function: measures how accurate the model is during the training. You want to minimize this function to steer the model in the right direction
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # Metrics: Monitor the training and testing steps.
    # The following uses "accuracy", the fraction of the images that are correctly classified
    metrics=['accuracy']
)

# Fit Keras Model
model.fit(train_imgs, train_labs, epochs=10)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_imgs, test_labs, verbose=2)
print('\nTest accuracy: ', test_acc)


# Make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_imgs)

# Graph this to look at the full set of 10 class prediction


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel('{} {:2.0f}% ({})'. format(class_names[predicted_label],
                                          100*np.max(predictions_array),
                                          class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Verify predictions
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labs, test_imgs)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labs)
plt.show()
