import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


## features
X = np.array([-7.0,-4.0,-1.0,2.0,5.0, 8.0, 11.0, 14.0])



##labels

y = np.array([3.0,6.0,9.0,12.0,15.0,18.0,21.0,24.0])

X = tf.constant(X)
y = tf.constant(y)

print(X,y)
plt.scatter(X,y)

plt.show()



input_shape = X[0].shape
output_shape = y[0].shape
print(input_shape,output_shape)

tf.random.set_seed(42)

## Create a model using the Sequential Api
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

## Complie the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

##Fit the model
model.fit(tf.expand_dims(X,axis=-1),y,epochs=50)

print(X)
print(y)

print(model.predict([17.0]))