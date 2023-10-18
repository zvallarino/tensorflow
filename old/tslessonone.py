import random

import tensorflow as tf
import numpy as np
#
# print(tf.__version__)
#
# scalar = tf.constant(7)
# print(scalar)
#
# print(scalar.ndim)
#
# vector = tf.constant([7,2])
#
# print(vector.ndim)
#
# matrix = tf.constant(([10,7],[6,10],[6,10]))
#
# print(matrix.shape)
# print(matrix.ndim)
#
#
# random_1 = tf.random.Generator.from_seed(42)
# random_1 = random_1.normal(shape=(3,2))
# random_2 = tf.random.Generator.from_seed(42)
# random_2 = random_2.normal(shape=(3,2))
#
# print(random_1)
# print(random_2)
# print(random_2==random_1)
#
#
# random_1 = tf.random.shuffle(random_1)
#
#
# print(random_2==random_1)
#
# #
# # ones = tf.ones([3,2])
# #
# # print(ones)
# #
# # zeros = tf.zeros([3,2,3])
# #
# # print(zeros)
# #
# # numpy_B = np.arange(1,41,dtype=np.int32)
# #
# # print(numpy_B)
# #
# # A = tf.constant(numpy_B,shape=(10,4,1))
# #
# # # print(A)
# # #
# # # zeros = tf.zeros([3,3,4,6,2])
# # #
# # # print("Datatype of every element:",zeros.dtype)
# # # print("Number of dimesions(rank)", zeros.ndim)
# # # print("Shape of tensor:",zeros.shape)
# # # print("Elements along the 0 axis",zeros.shape[0])
# # # print('size',tf.size(zeros).numpy())
# # #
# # #
# # # rank_2_tensor = tf.ones([3,2])
# # # print(rank_2_tensor)
# # #
# # # rank_3_tensor = rank_2_tensor[...,tf.newaxis]
# # # print(rank_3_tensor.shape)
# #
# # matrix_one = tf.constant([[1,2],
# #                           [4,5],
# #                          [7,8]])
# #
# # matrix_two = tf.constant([[8,6],
# #                           [3,4],
# #                          [2,3]])
# #
# # three = tf.constant(
# #     [[
# #         [1,2],
# #         [4,5]
# #     ],
# #                          [
# #                              [7,8],
# #                              [9,10]
# #                          ]
# #     ])
# #
# # print(matrix_one.shape)
# # print(matrix_two.shape)
# # print(three.shape)
#
# # print(tf.matmul(matrix_two,three))
#
# z = tf.constant([
#     [2,3],
#     [4,5],
#     [7,10],
# ], dtype=tf.float32)
#
#
# print(z)
# min_val = tf.math.reduce_min(z)
# max_val = tf.math.reduce_max(z)
# mean_val = tf.math.reduce_mean(z)
# sum_val = tf.math.reduce_sum(z)
#
# print("Minimum:", min_val.numpy())
# print("Maximum:", max_val.numpy())
# print("Mean:", mean_val.numpy())
# print("Sum:", sum_val.numpy())
#
# # Compute variance and standard deviation
# variance = tf.math.reduce_variance(z)
# std_dev = tf.math.reduce_std(z)
#
# # Print the values
# print("Variance:", variance.numpy())
# print("Standard Deviation:", std_dev.numpy())
# tf.random.set_seed(42)
# g = tf.constant(tf.random.uniform(shape=[60]),shape=(1,2,15,2))
#
# print(g)
#
# some_list = [1,2,3,4]
#
# hots = tf.one_hot(some_list,depth=4, on_value="blue", off_value="red")
# print(tf.config.list_physical_devices())
#
# scalar_A = tf.constant(7)
# vector_A = tf.constant([10])
# matrix_A = tf.constant([[2,3],[4,1]])
# tensor_A = tf.constant([[[1,2,3],[4,5,6]],
#                         [[4,6,8],[10,12,14]]])
#
# def functionS(matrixA):
#     print("--------")
#     print ("Its size:", tf.size(matrixA).numpy())
#     print("Its rank:", matrixA.ndim)
#     print ("Its shape:", matrixA.shape)
#     print("--------")
#     return
#
#
# functionS(scalar_A)
# functionS(vector_A)
# functionS(matrix_A)
# functionS(tensor_A)
#
# matrix_one = tf.random.uniform(shape=(5, 300), minval=0, maxval=1)
# matrix_two = tf.random.uniform(shape=(5, 300), minval=0, maxval=1)
#
# print(matrix_one.shape)
# print(matrix_one*matrix_two)
#
# print(tf.matmul(tf.transpose(matrix_one), matrix_two))
# matrix_five = tf.matmul(matrix_one, tf.transpose(matrix_two))
# matrix_four = tf.matmul(tf.transpose(matrix_one), matrix_two)
#
# print(matrix_five.shape)
#
# matrix_three = tf.random.uniform(shape=(224, 224, 3), minval=0, maxval=1)
#
# min_val = tf.math.reduce_min(matrix_three)
# max_val = tf.math.reduce_max(matrix_three)
# mean_val = tf.math.reduce_mean(matrix_three)
# sum_val = tf.math.reduce_sum(matrix_three)
#
# print("Minimum:", min_val.numpy())
# print("Maximum:", max_val.numpy())
# print("Mean:", mean_val.numpy())
# print("Sum:", sum_val.numpy())
#
# matrix_three = tf.random.uniform(shape=(1, 224, 224, 3), minval=0, maxval=1)
#
# squeezed = tf.squeeze(matrix_three,axis=0)

print(squeezed.shape)

matrix_four = tf.random.uniform(shape=[10], minval=0, maxval=10)

print(matrix_four)
print(tf.argmax(matrix_four))

print(matrix_four[tf.argmax(matrix_four)])

print(tf.one_hot(tf.cast(matrix_four,dtype='int32'),depth=10))