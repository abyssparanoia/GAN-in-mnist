import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


def tensor_to_vector(input):
    shape = input.get_shape()[1:].as_list()
    dim = np.prod(shape)
    return tf.reshape(input, [-1, dim]), dim


def leaky_relu(input):
    return tf.maximum(0.2 * input, input)


def visualize(images, num_itr, rows, cols):
    plt.title(num_itr, color="red")
    for index, data in enumerate(images):
        plt.subplot(rows, cols, index + 1)
        plt.axis("off")
        plt.imshow(data.reshape(28, 28), cmap="gray", interpolation="nearest")
    plt.show()