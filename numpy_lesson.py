import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd

## numpy's main data type is a ndarray

## an array of however many dimensions

a1 = np.array([1,2,3])


a2 = np.array([[1.,2.0,3.3],[4.,5.,6.]])



a3 = np.array([
            [[1,2,3],
               [4,5,6],
               [7,8,9]],
              [[10,11,12],
               [13,14,15],
               [15,16,17]]]
              )

# print(a1.shape)
#
# # a1 it stores it without the row number so its a 1,3 but when you print it, it says 3,
#
# #all the other ones are correct
#
# print(a2.shape)
# print(a3.shape)
#
# print(a1.ndim,a2.ndim,a3.ndim)
# print(a1.size,a2.size,a3.size)
# print(a1.size,a2.size,a3.size)

# sample_array = np.array([1,2,3])
#
# print(sample_array)
#
# print(type(sample_array))
#
# ones = np.ones((2,3))
# print(ones.dtype)
#
# zeros = np.zeros((2,3))
#
# print(zeros)
#
# range_array = np.arange(0,20,3)
#
# print(range_array)
#
# random_array = np.random.randint(0,12,size=(3,5))
#
# print(random_array)
#
# print(random_array.shape)
#
# random_array_2 = np.random.random((5,2))
#
# print(random_array_2)
#
# random_array_3 = np.random.rand(5,2)
#
# print(random_array_3)
# np.random.seed(seed=0)
# random_array_4 = np.random.randint(12,size=(6,2))
#
# print(random_array_4)
#
# print(random_array_4)

# massive_array = np.random.random(10000)
#
# print(massive_array.size)



#
#
# # Start the timer
# start_time = time.time()
#
# # The code you want to measure
# sum_of_array = sum(massive_array)
#
# # Stop the timer
# end_time = time.time()
#
# # Calculate and print the elapsed time
# elapsed_time = end_time - start_time
# print(f"Time taken: {elapsed_time:.6f} seconds")
#
# start_time = time.time()
#
# # The code you want to measure
# sum_of_array = np.sum(massive_array)
#
# # Stop the timer
# end_time = time.time()
#
# # Calculate and print the elapsed time
# elapsed_time = end_time - start_time
# print(f"Time taken: {elapsed_time:.6f} seconds")
#


np.std(a2)


##variance = the measure of the average degree to which each number is different to the mean

##higher variance = higher range of numbers

##lower variance = lower range of number

##standard deviation is equal to the squared root of variance

standard_variation = np.sqrt(np.var(a2))



# high_variance_array = np.array([1,10000,200000,30000000,50000000])
#
# low_variance_array = np.array([1,2,3,4,5])
#
# print(np.var(high_variance_array))
# print(np.var(low_variance_array))
#
#
# high_standard_variation = np.sqrt(np.var(high_variance_array))
# low_standard_variation = np.sqrt(np.var(low_variance_array))
# print(high_standard_variation)
# print(np.std(high_variance_array))
# print(np.std(low_variance_array))
# print(low_standard_variation)
#
# plt.hist(low_variance_array)
# plt.show()

print(a2.shape)

print(a3.shape)


## in broadcasting, multiplcation can happen when the shapes are equal or one the shapes is one. Can you explain that with a short demo?


np.random.seed(0)
array_data = np.random.randint(1, 21, size=(5, 3))

print(array_data)

tableNuts = pd.DataFrame(array_data,index=['Mon',"Tue",'Wed','Thu','Friday'],columns=['Almond Butter','Peanut Butter','Cashew Butter'])

print(tableNuts)

prices = np.array([10, 8, 12])


priceTable = pd.DataFrame(prices.reshape(1,3), index=['Price'], columns=['Almond Butter','Peanut Butter','Cashew Butter'])

print(priceTable)

tableNuts_T = tableNuts.T

totals = np.dot(priceTable,tableNuts.T)

print(totals)

tableNuts["totals"] = totals.T

print(tableNuts)

ai_photo = imread("C:\\Users\\zvallarino\\OneDrive - The Population Council, Inc\\Documents\\DataForPanda\\gats.jpeg")

print(ai_photo.size,ai_photo.shape,ai_photo.ndim)

print(ai_photo[:5])

couple = imread("C:\\Users\\zvallarino\\OneDrive - The Population Council, Inc\\Documents\\DataForPanda\\about_2.jpg")

print(couple.size,couple.shape,couple.ndim)

print(couple[:5])