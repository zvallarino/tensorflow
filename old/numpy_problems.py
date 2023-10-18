import numpy as np
import pandas as pd

one_d = np.random.randint(1,10,size=(3))
two_d = np.random.randint(1,10,size=(2,3))
three_d = np.random.randint(1,10,size=(2,3,3))


#
# print(one_d.shape, one_d.ndim, one_d.dtype, type(one_d))
#
# print(two_d.shape, two_d.ndim, two_d.dtype, type(two_d))
#
# print(three_d.shape, three_d.ndim, three_d.dtype, type(three_d))
#
# print(one_d)
# one_t = pd.DataFrame([one_d], index=['cats'], columns = ['red','blue','white'])
#
# print(one_t)
#
# ones = np.ones((10,2))
#
# print(ones.shape)
#
# print(ones)
#
# zeros = np.zeros((7,2,3))
#
# print(zeros.shape)
#
# array_new = np.arange(0,100,3)
# print(array_new)
#
# next_arr=np.random.uniform(0, 1, size=(3, 5))
#
# print(next_arr)
#
# np.random.seed(seed=42)
#
# nn_arr = np.random.randint(0,10,size=(4,6))
#
# print(nn_arr)
# print(nn_arr.shape)

# new_way = np.random.randint(1,10,size=(3,5))
#
# ones = np.ones((3,5))
#
# added = new_way + ones
#
#
# onesT = ones.T
#
# # added_atttempt = new_way + onesT
#
# newer_array = added - ones
#
# mathM = newer_array * ones
# meaner = mathM.mean()
# print(np.var(mathM))
# reshapes = np.reshape(mathM,newshape=(3,5,1))
#
# print(reshapes.T)
# #
# print(new_way)
# print('-------')
#
# print(new_way[:2, :2])

x = np.random.randint(0,10,size=(3,2))

y = np.random.randint(0,10,size=(3,2))

# z = np.dot(x,y)
#
# print(z)
#
# u = np.random.randint(0,10,size=(4,3))
#
# v = np.random.randint(0,10,size=(4,3))
#
# zz = np.dot(u,v.T)
#
# print(zz)

print(x)



normal_array = np.linspace(1, 100, 10)

dif = np.

print(normal_array)