import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
# print(tf.__version__)
#
#
## features
# X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
#
#
#
# ##labels
#
# y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
#
# X = tf.range(-100, 100, 4)
#
#
# y = X + 10
#
# print(len(X))
#
# # 80% fo the data
# x_train = X[:40]
# # 20% of the data
# x_test = X[40:]
# # train_data
#
# y_train = y[:40]
# y_test = y[40:]
# print(x_test)
# print(y_test)
#
# print(len(x_train),len(x_test),len(y_train),len(y_test))
#
# plt.figure(figsize=(7,10))
# plt.scatter(x_train,y_train,c="b",label="training data")


#
# x_test=
# test_data =
# print(X,y)
# plt.scatter(X,y)
#
# plt.show()
#
#
#
# input_shape = X[0].shape
# output_shape = y[0].shape
# print(input_shape,output_shape)
#
# tf.random.set_seed(42)
#
# ## Create a model using the Sequential Api
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(1)
# ])
#
# ## Complie the model
# model.compile(loss=tf.keras.losses.mae,
#               optimizer=tf.keras.optimizers.SGD(),
#               metrics=["mae"])
#
# ##Fit the model
# model.fit(tf.expand_dims(X,axis=-1),y,epochs=100)
#
#
# print(model.predict([17.0]))
#
# # 1. Create the model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(100, activation="relu"),
#     tf.keras.layers.Dense(1)
# ])
#
# # 2. Complie the model
# model.compile(loss=tf.keras.losses.mae,
#               optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
#               metrics=['mae'])
#
# #3. Fit the model(this time we will train for longer)
# model.fit(tf.expand_dims(X,axis=-1),y, epochs=200)
#
# plt.scatter(X,y)
# plt.show()
#
# print(model.predict([17.0]))


#

#
# # step one create a model
# model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
#
# #step two complie the model
#
# model.compile(loss=tf.keras.losses.mae,
#               optimizer=tf.keras.optimizers.SGD(),
#               metrics=["mae"])
#
# #step three fit the model
# # model.fit(x_train,y_train,epochs=100)
#
# model.summary()


X = tf.range(-100, 100, 4)


y = X + 10

print(len(X))

# 80% fo the data
x_train = X[:40]
# 20% of the data
x_test = X[40:]
# train_data

y_train = y[:40]
y_test = y[40:]
print(x_test)
print(y_test)

print(len(x_train),len(x_test),len(y_train),len(y_test))

plt.figure(figsize=(7,10))
plt.scatter(x_train,y_train,c="b",label="training data")
#lets create a model and build it automatically by defining the input shape

# tf.random.set_seed(42)
#
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10,input_shape=[1], name="input_layer"),
    tf.keras.layers.Dense(1, name='output_layer')

],name ='model_1')

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])
#
model.fit(tf.expand_dims(x_train,axis=-1),y_train, epochs=100, verbose=0)
#
y_pred = model.predict(x_test)
#
# print(x_test)
# print(y_pred)

def plottingfunction(
        train_data=x_train,
        train_labels=y_train,
        test_data=x_test,
        test_labels=y_test,
        predictions=y_pred):
    plt.figure(figsize=(7,10))
    plt.scatter(train_data,train_labels,c="blue")
    plt.scatter(test_data,test_labels,c='red')
    plt.scatter(test_data,predictions,c='green')
    plt.show()



def mae(y_true,y_pred):
    return tf.metrics.mean_absolute_error(y_true=y_true,y_pred=y_pred)

def mse(y_true,y_pred):
    return tf.metrics.mean_squared_error(y_true=y_true,y_pred=y_pred)

tf.random.set_seed(42)
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(10,input_shape=[1], name="input_layer"),
#     tf.keras.layers.Dense(1, name='output_layer')
#
# ],name ='model_1')
#
# model.compile(loss=tf.keras.losses.mae,
#               optimizer=tf.keras.optimizers.SGD(),
#               metrics=['mae'])
#
# model.fit(tf.expand_dims(x_train,axis=-1),y_train, epochs=100, verbose=0)
#
# y_pred = model.predict(x_test)
#
# print(x_test)
# print(y_pred)

#model 1

model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

model_1.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

model_1.fit(tf.expand_dims(x_train,axis=-1),y_train,epochs=100)

y_pred_1 = model_1.predict(x_test)

model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

model_2.fit(tf.expand_dims(x_train,axis=-1),y_train,epochs=100)

y_pred_2 = model_2.predict(x_test)

model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

model_3.fit(tf.expand_dims(x_train, axis=-1), y_train, epochs=200)

y_pred_3 = model_3.predict(x_test)

print(y_pred_1,y_pred_2,y_pred_3)

plottingfunction(predictions=y_pred_1)
plottingfunction(predictions=y_pred_2)
plottingfunction(predictions=y_pred_3)
