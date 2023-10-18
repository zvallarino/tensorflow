import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
#
# insurance = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
#
# insurance_encoded = pd.get_dummies(insurance)
# #
#
# X = insurance_encoded.drop('charges', axis=1).astype('float32')
# y = insurance_encoded['charges'].astype('float32')
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# print(len(X),len(X_train),len(X_test))
# #
# #
# #
# # pd.set_option('display.max_columns', None)
# #
# #
# insurance_model = tf.keras.Sequential([
#     tf.keras.layers.Dense(32, activation='relu', input_shape=(11,)),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(1, name='output_layer')# Output layer
# ], name='baseline_model')
#
# insurance_model.compile(loss=tf.keras.losses.mae,
#               optimizer=tf.keras.optimizers.Adam(lr=0.001),
#               metrics=['mae'])
# #
# history = insurance_model.fit(tf.expand_dims(X_train,axis=-1),y_train, epochs=2000)
# # I think its called like a loss curve
#
# y_pred = insurance_model.predict(X_test)
#
# pd.DataFrame(history.history).plot()
# plt.ylabel('loss')
# plt.xlabel('epochs')
#
# plt.show()

#
# def plottingfunction(
#         train_data=X_train,
#         train_labels=y_train,
#         test_data=X_test,
#         test_labels=y_test,
#         predictions=y_pred):
#     plt.figure(figsize=(10, 7))
#
#     # Use a range for x-axis
#     plt.scatter(range(len(train_labels)), train_labels, c="blue", label="Training Data")
#     plt.scatter(range(len(train_labels), len(train_labels) + len(test_labels)), test_labels, c="red", label="Test Data")
#     plt.scatter(range(len(train_labels), len(train_labels) + len(test_labels)), predictions, c="green", label="Predictions")
#
#     plt.legend()
#     plt.show()

# def mae(y_true,y_pred):
#     return tf.metrics.mean_absolute_error(y_true=y_true,y_pred=tf.squeeze(y_pred))
#
# def mse(y_true,y_pred=y_pred):
#     return tf.metrics.mean_squared_error(y_true=y_true,y_pred=tf.squeeze(y_pred))
#
# print('Mae', mae(y_test,y_pred))
# print('Mse', mse(y_test,y_pred))

# plottingfunction(predictions=y_pred)

# tf.random.set_seed(42)
#
# model_2 = tf.keras.Sequential([
#     tf.keras.layers.Dense(10),
#     tf.keras.layers.Dense(1),
# ])
#
# model_2.compile(loss=tf.keras.losses.mae,
#                 optimizer=tf.keras.optimizers.SGD(),
#                 metrics=['mse'])
#
# model_2.fit(tf.expand_dims(x_train,axis=-1),y_train,epochs=100,verbose=0)
#
# y_pred_2 = model_2.predict(x_test)
#
#
# tf.random.set_seed(42)

insurance = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
pd.set_option('display.max_columns', None)
print(insurance)

##for age.bmi and children they are numbers but they should be between 0-1

ct = make_column_transformer(
    (MinMaxScaler(), ['age', 'bmi', 'children']),
    (OneHotEncoder(handle_unknown='ignore'), ['sex', 'smoker', 'region'])##handle ingore tells onehotencoder to ingore everything not listed
)

X = insurance.drop('charges',axis=1)
y = insurance['charges']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#fit the column transformer to our training data
ct.fit(X_train)
x_train_normal = ct.transform(X_train)
x_test_normal = ct.transform(X_test)

print(X_train.loc[0])
print(x_train_normal[0])

# Adjust the input shape based on the transformed data
input_shape = x_train_normal.shape[1]

insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, name='output_layer')  # Output layer
], name='baseline_model')

insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        metrics=['mae'])

history = insurance_model.fit(x_train_normal, y_train, epochs=2000)

# Plot the loss curve
pd.DataFrame(history.history).plot()
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()





