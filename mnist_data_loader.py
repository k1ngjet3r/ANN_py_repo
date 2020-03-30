import numpy as np
import matplotlib.pyplot as plt
import random

image_size = 28  # width and length
num_diff_labs = 10  # i.e. 0, 1, 2, 3, ..., 9
image_pixs = image_size * image_size
data_path = "/Users/jeter/Documents/mnist/"
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

# The images of the MNIST dataset are greyscale and the pixels range
# between 0 and 255 including both bounding values. We will map these
# values into an interval from [0.01, 1] by multiplying each pixel
# by 0.99 / 255 and adding 0.01 to the result. This way, we avoid 0
# values as inputs, which are capable of preventing weight updates,
# as we we seen in the introductory chapter.
frc = 0.99 / 255
test_imgs = np.asfarray(test_data[:, 1:], dtype="float") * frc + 0.01
train_imgs = np.asfarray(train_data[:, 1:], dtype="float") * frc + 0.01

test_labs = np.asfarray(test_data[:, :1], dtype="float")
train_labs = np.asfarray(train_data[:, :1], dtype="float")

lr = np.arange((num_diff_labs))

test_labs_onehot = (lr == test_labs).astype(np.float)
train_labs_onehot = (lr == train_labs).astype(np.float)

# turn our labelled images into one-hot representations. Instead
# of zeroes and one, we create 0.01 and 0.99, which will be better
# for our calculations:
test_labs_onehot[test_labs_onehot == 0] = 0.01
test_labs_onehot[test_labs_onehot == 1] = 0.99
train_labs_onehot[train_labs_onehot == 0] = 0.01
train_labs_onehot[train_labs_onehot == 1] = 0.99

test_input = list(zip(test_labs_onehot, test_imgs))
train_input = list(zip(train_labs_onehot, train_imgs))


# # below is the code to show the data in picture form
# for i in range(10):
#     img = test_imgs[i].reshape((28, 28))
#     plt.imshow(img, cmap="Greys")
#     plt.show()
