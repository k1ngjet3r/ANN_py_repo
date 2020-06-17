# Determine the reviews posted on the IMDB website were positive or negative
# The dataset contains the text of 50,000 movie reviews from IMDB, and split into 25,000
# reviews for training and 25,000 for testing. The training and testing sets are balanced
# meaning that they contain an equal number of positive and negative reviews.

import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf

# download the IMDB dataset
train_data, validation_data, test_data = tfds.load(
    name='imdb_reviews',
    # split the training set into 60% and 40%, thus, we have 15,000 training,
    # 10,000 valication examples and 25,000 testing examples
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)

# print first 10 examples
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

# Build the model
# This is a pre-trained text embedding model from Tensorflow Hub
# Use this pre-trained text embedding as the first layers
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
# Note no matter the length of the input text, the output shape of the embeddings is
# (num_examples, embedding_dimension)

# Build the full model
model = tf.keras.Sequential()
# 1st layer: a TensorFlow Hub layer, uses a pre-trained Saved model to map a sectence
# into its embedding vector and split the sentence into tokens, embedds each token and
# then combines the embedding.
model.add(hub_layer)
# 2nd layer: fixed-length output vector is piped through a fully-connected(Dense)
# layer with 16 hidden units.
model.add(tf.keras.layers.Dense(16, activation='relu'))
# Last layer: densely connected with a single output node.
model.add(tf.keras.layers.Dense(1))

model.summary()

# Compile the model
model.compile(optimizer='adam',
              # Since this is a binary classsification problem and the model output a
              # probability (a single-unit layer with a sigmoid activation), we'll use
              # the binary_crossentropy loss function.
              loss=tf.keras.losses.BinaryCrossentropy(
                  from_logits=True), metrics=['accuracy'])

# Train the model
# train the model for 20 epochs in mini-batches of 512 samples. this is 20 iterations over all samples
# in the x_train and y_train tensors. while training, monitor the model's loss and accuracy on the
# 10,000 samples from the validation set.
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# Evaluate the model

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print('%s: %.3f' % (name, value))
