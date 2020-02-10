#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing
import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.__version__


# In[2]:


mnist = input_data.read_data_sets("")


# In[3]:


train_X = mnist.train._images.reshape((-1, 28, 28, 1))
train_Y = mnist.train._labels
train_X.shape


# In[4]:


test_X = mnist.test._images.reshape((-1, 28, 28, 1))
test_Y = mnist.test._labels
test_X.shape


# In[5]:


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(dataset, labels, name, i):
    if not os.path.exists(name):
        os.mkdir(name)
    filename = os.path.join(name, "file-%d.tfrecords" % (i))
    print("writing %s, cpu %d" % (filename, i))
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(dataset.shape[0]):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": _int64_feature(int(labels[index])),
                        "image": _bytes_feature(dataset[index].tostring()),
                    }
                )
            )
            writer.write(example.SerializeToString())


# In[6]:


cpu_cores = 2
train_idx = np.linspace(0, train_X.shape[0], cpu_cores + 1, dtype=np.int)
test_idx = np.linspace(0, test_X.shape[0], cpu_cores + 1, dtype=np.int)
pool = multiprocessing.Pool(processes=cpu_cores)

for p in range(cpu_cores):
    pool.apply_async(
        convert_to,
        (
            train_X[train_idx[p] : train_idx[p + 1] - 1],
            train_Y[train_idx[p] : train_idx[p + 1] - 1],
            "train",
            p,
        ),
    )

for p in range(cpu_cores):
    pool.apply_async(
        convert_to,
        (
            test_X[train_idx[p] : test_idx[p + 1] - 1],
            test_Y[train_idx[p] : test_idx[p + 1] - 1],
            "test",
            p,
        ),
    )

pool.close()
pool.join()


# In[7]:


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        },
    )
    image = tf.decode_raw(features["image"], tf.float32)
    image = tf.reshape(image, [28, 28, 1])
    image = tf.image.per_image_standardization(image)
    label = tf.cast(features["label"], tf.int32)
    return image, label


# In[8]:


tf.reset_default_graph()
sess = tf.InteractiveSession()

batch_size = 128
epoch = 5
filename_queue = tf.train.string_input_producer(
    ["train/" + i for i in os.listdir("train")], num_epochs=epoch
)
image, label = read_and_decode(filename_queue)
images, labels = tf.train.shuffle_batch(
    [image, label],
    batch_size=batch_size,
    num_threads=12,
    capacity=train_X.shape[0],
    min_after_dequeue=1000,
    allow_smaller_final_batch=False,
)


def convolutionize(x, conv_w, h=1):
    return tf.nn.conv2d(input=x, filter=conv_w, strides=[1, h, h, 1], padding="SAME")


def pooling(wx):
    return tf.nn.max_pool(wx, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def create_network(X, scope="conv", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        w1 = tf.Variable(tf.random_normal([3, 3, 1, 2], stddev=0.5))
        b1 = tf.Variable(tf.zeros(shape=[2]))
        w2 = tf.Variable(tf.random_normal([3, 3, 2, 4], stddev=0.5))
        b2 = tf.Variable(tf.zeros(shape=[4]))
        w3 = tf.Variable(tf.random_normal([3, 3, 4, 8], stddev=0.5))
        b3 = tf.Variable(tf.zeros(shape=[8]))
        w4 = tf.Variable(tf.random_normal([128, 10], stddev=0.5))
        b4 = tf.Variable(tf.zeros(shape=[10]))

        conv1 = pooling(tf.nn.relu(convolutionize(X, w1) + b1))
        conv2 = pooling(tf.nn.relu(convolutionize(conv1, w2) + b2))
        conv3 = pooling(tf.nn.relu(convolutionize(conv2, w3) + b3))
        conv3 = tf.reshape(conv3, [-1, 128])
        return tf.matmul(conv3, w4) + b4


logits = create_network(images)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
global_step = tf.Variable(0, name="global_step", trainable=False)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost, global_step=global_step)
correct_pred = tf.equal(tf.argmax(logits, 1, output_type=tf.int32), labels)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)


# In[9]:


coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:
    step = sess.run(global_step)
    while not coord.should_stop():
        acc, loss, _, _, _ = sess.run([accuracy, cost, optimizer, images, labels])
        if step % 200 == 0:
            print("step %d, loss %f, accuracy %f" % (step, loss, acc))
        step = sess.run(global_step)
except tf.errors.OutOfRangeError:
    print("Done training for %d epochs, %d steps." % (epoch, step))
finally:
    coord.request_stop()
coord.join(threads)


# In[ ]:
