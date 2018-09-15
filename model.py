import tensorflow as tf
import numpy as np
from utils import tensor_to_vector, leaky_relu


class MNIST_DCGAN():
    def __init__(
            self,
            batch_size,
            image_shape=[28, 28, 1],
            dim_z=100,
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_ch=1,
    ):
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_ch = dim_ch

        # variable of generator
        self.g_W1 = tf.Variable(
            tf.random_normal([dim_z, dim_W1], stddev=0.02), name='g_W1')
        self.g_b1 = tf.Variable(
            tf.random_normal([dim_W1], stddev=0.02), name='g_b1')
        self.g_W2 = tf.Variable(
            tf.random_normal([dim_W1, dim_W2 * 7 * 7], stddev=0.02),
            name='g_W2')
        self.g_b2 = tf.Variable(
            tf.random_normal([dim_W2 * 7 * 7], stddev=0.02), name='g_b2')
        self.g_W3 = tf.Variable(
            tf.random_normal([5, 5, dim_W3, dim_W2], stddev=0.02), name="g_W3")
        self.g_b3 = tf.Variable(
            tf.random_normal([dim_W3], stddev=0.02), name="g_b3")
        self.g_W4 = tf.Variable(
            tf.random_normal([5, 5, dim_ch, dim_W3], stddev=0.02), name="g_W4")
        self.g_b4 = tf.Variable(
            tf.random_normal([dim_ch], stddev=0.02), name="g_b4")

        # variable of discriminator
        self.d_W1 = tf.Variable(
            tf.random_normal([5, 5, dim_ch, dim_W3], stddev=0.02), name="d_W1")
        self.d_b1 = tf.Variable(
            tf.random_normal([dim_W3], stddev=0.02), name="d_b1")
        self.d_W2 = tf.Variable(
            tf.random_normal([5, 5, dim_W3, dim_W2], stddev=0.02), name="d_W2")
        self.d_b2 = tf.Variable(
            tf.random_normal([dim_W2], stddev=0.02), name="d_b2")
        self.d_W3 = tf.Variable(
            tf.random_normal([dim_W2 * 7 * 7, dim_W1], stddev=0.02),
            name="d_W3")
        self.d_b3 = tf.Variable(
            tf.random_normal([dim_W1], stddev=0.02), name="d_b3")
        self.d_W4 = tf.Variable(
            tf.random_normal([dim_W1, 1], stddev=0.02), name="d_W4")
        self.d_b4 = tf.Variable(
            tf.random_normal([1], stddev=0.02), name="d_b4")

    def build(self):
        Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])
        img_real = tf.placeholder(tf.float32,
                                  [self.batch_size] + self.image_shape)
        img_gen = self.generate(Z)

        raw_real = self.discriminate(img_real)
        raw_gen = self.discriminate(img_gen)

        p_real = tf.nn.sigmoid(raw_real)
        p_gen = tf.nn.sigmoid(raw_gen)

        discrim_cost = tf.reduce_mean(
            -tf.reduce_sum(tf.log(p_real) + \
                           tf.log(tf.ones(self.batch_size, tf.float32) - p_gen), axis=1))
        gen_cost = tf.reduce_mean(-tf.reduce_sum(tf.log(p_gen), axis=1))

        return Z, img_real, discrim_cost, gen_cost, p_real, p_gen

    def generate(self, Z):
        #1st layer
        fc1 = tf.matmul(Z, self.g_W1) + self.g_b1
        bm1, bv1 = tf.nn.moments(fc1, axes=[0])
        bn1 = tf.nn.batch_normalization(fc1, bm1, bv1, None, None, 1e-5)
        relu1 = tf.nn.relu(bn1)

        #2nd layer
        fc2 = tf.matmul(relu1, self.g_W2) + self.g_b2
        bm2, bv2 = tf.nn.moments(fc2, axes=[0])
        bn2 = tf.nn.batch_normalization(fc2, bm2, bv2, None, None, 1e-5)
        relu2 = tf.nn.relu(bn2)

        y2 = tf.reshape(relu2, [self.batch_size, 7, 7, self.dim_W2])

        #3rd layer
        conv_t1 = tf.nn.conv2d_transpose(
            y2,
            self.g_W3,
            strides=[1, 2, 2, 1],
            output_shape=[self.batch_size, 14, 14, self.dim_W3]) + self.g_b3
        bm3, bv3 = tf.nn.moments(conv_t1, axes=[0, 1, 2])
        bn3 = tf.nn.batch_normalization(conv_t1, bm3, bv3, None, None, 1e-5)
        relu3 = tf.nn.relu(bn3)

        #4th layer
        conv_t2 = tf.nn.conv2d_transpose(
            relu3,
            self.g_W4,
            strides=[1, 2, 2, 1],
            output_shape=[self.batch_size, 28, 28, self.dim_ch]) + self.g_b4
        img = tf.nn.sigmoid(conv_t2)

        return img

    def discriminate(self, img):
        #1st layer
        conv1 = tf.nn.conv2d(
            img, self.d_W1, strides=[1, 2, 2, 1], padding='SAME')
        y1 = leaky_relu(conv1)

        #2nd layer
        conv2 = tf.nn.conv2d(
            y1, self.d_W2, strides=[1, 2, 2, 1], padding="SAME") + self.d_b2
        y2 = leaky_relu(conv2)

        #3rd layer
        vec, _ = tensor_to_vector(y2)
        fc1 = tf.matmul(vec, self.d_W3) + self.d_b3
        y3 = leaky_relu(fc1)

        #4th layer
        fc2 = tf.matmul(y3, self.d_W4) + self.d_b4

        return fc2

    def generate_samples(self, batch_size):
        Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])

        fc1 = tf.matmul(Z, self.g_W1) + self.g_b1
        bm1, bv1 = tf.nn.moments(fc1, axes=[0])
        bn1 = tf.nn.batch_normalization(fc1, bm1, bv1, None, None, 1e-5)
        relu1 = tf.nn.relu(bn1)

        fc2 = tf.matmul(relu1, self.g_W2) + self.g_b2
        bm2, bv2 = tf.nn.moments(fc2, axes=[0])
        bn2 = tf.nn.batch_normalization(fc2, bm2, bv2, None, None, 1e-5)
        relu2 = tf.nn.relu(bn2)

        y2 = tf.reshape(relu2, [batch_size, 7, 7, self.dim_W2])

        conv_t1 = tf.nn.conv2d_transpose(
            y2,
            self.g_W3,
            strides=[1, 2, 2, 1],
            output_shape=[batch_size, 14, 14, self.dim_W3]) + self.g_b3
        bm3, bv3 = tf.nn.moments(conv_t1, axes=[0, 1, 2])
        bn3 = tf.nn.batch_normalization(conv_t1, bm3, bv3, None, None, 1e-5)
        relu3 = tf.nn.relu(bn3)

        conv_t2 = tf.nn.conv2d_transpose(
            relu3,
            self.g_W4,
            strides=[1, 2, 2, 1],
            output_shape=[batch_size, 28, 28, self.dim_ch]) + self.g_b4
        img = tf.nn.sigmoid(conv_t2)
        return Z, img
