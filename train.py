import tensorflow as tf
import numpy as np
from model import MNIST_DCGAN
from utils import visualize
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def train(train_imgs, n_epochs, batch_size):

    dcgan_model = MNIST_DCGAN(batch_size=128, image_shape=[28, 28, 1])
    Z_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen = dcgan_model.build()
    Z_gen, image_gen = dcgan_model.generate_samples(batch_size=32)

    discrim_vars = [x for x in tf.trainable_variables() if "d_" in x.name]
    gen_vars = [x for x in tf.trainable_variables() if "g_" in x.name]

    optimizer_d = tf.train.AdamOptimizer(
        0.0002, beta1=0.5).minimize(
            d_cost_tf, var_list=discrim_vars)
    optimizer_g = tf.train.AdamOptimizer(
        0.0002, beta1=0.5).minimize(
            g_cost_tf, var_list=gen_vars)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=10)

    tf.global_variables_initializer().run()

    itr = 0
    for epoch in range(n_epochs):
        index = np.arange(len(train_imgs))
        np.random.shuffle(index)
        trX = train_imgs[index]

        for start, end in zip(
                range(0, len(trX), batch_size),
                range(batch_size, len(trX), batch_size)):
            Xs = trX[start:end].reshape([-1, 28, 28, 1]) / 255.
            Zs = np.random.uniform(
                -1, 1, size=[batch_size, dcgan_model.dim_z]).astype(np.float32)

            if np.mod(itr, 2) != 0:
                _, gen_loss_val = sess.run(
                    [optimizer_g, g_cost_tf], feed_dict={Z_tf: Zs})
                discrim_loss_val, p_real_val, p_gen_val = sess.run(
                    [d_cost_tf, p_real, p_gen],
                    feed_dict={
                        Z_tf: Zs,
                        image_tf: Xs
                    })
                print("=========== updating G ==========")
                print("iteration:", itr)
                print("gen loss:", gen_loss_val)
                print("discrim loss:", discrim_loss_val)

            else:
                _, discrim_loss_val = sess.run(
                    [optimizer_d, d_cost_tf],
                    feed_dict={
                        Z_tf: Zs,
                        image_tf: Xs
                    })
                gen_loss_val, p_real_val, p_gen_val = sess.run(
                    [g_cost_tf, p_real, p_gen],
                    feed_dict={
                        Z_tf: Zs,
                        image_tf: Xs
                    })
                print("=========== updating D ==========")
                print("iteration:", itr)
                print("gen loss:", gen_loss_val)
                print("discrim loss:", discrim_loss_val)

            print("Average P(real)=", p_real_val.mean())
            print("Average P(gen)=", p_gen_val.mean())
            itr += 1

        z = np.random.uniform(
            -1, 1, size=[32, dcgan_model.dim_z]).astype(np.float32)
        generated_samples = sess.run([image_gen], feed_dict={Z_gen: z})
        visualize(generated_samples[0], epoch, 4, 8)
        print("epoch = ", epoch)


if __name__ == '__main__':
    train(mnist.test.images, 500, 128)
