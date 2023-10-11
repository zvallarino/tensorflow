import pandas as pd
import matplotlib.pyplot as plt
#
# colors = pd.Series(["Green","Red ","Blue"])
#
# # print(colors)
#
# makes = pd.Series(["Eco","Gundam",'Regular'])
#
# # print(makes)
#
# cars = pd.DataFrame({"makes":makes, "colors":colors})

# print(cars)
#
cars_csv = pd.read_csv('C:\\Users\\zvallarino\\OneDrive - The Population Council, Inc\\Documents\\DataForPanda\\car-sales.csv')
#
#
# print(cars_csv)
#
# # cars_csv.to_csv("C:/Users/zvallarino/OneDrive - The Population Council, Inc/Documents/DataForPanda/export_cars105.csv", index=False)
#
#
#
# print(cars_csv.dtypes)
#
# print(cars_csv.describe())
#
# print(cars_csv.info())
#
# series_of_num = pd.Series([6,4,3,2,1,10,23,4])
#
# print(series_of_num.mean())
#
# print(cars_csv.columns)
#
# # print(len(cars_csv))
#
# print(cars_csv.tail())
#
# print(cars_csv.head(7))
#
# print(cars_csv.loc[2])
#
# print(cars_csv.iloc[2])

# odometer_col = cars_csv[cars_csv['Odometer (KM)']>100000]
#
#
# ct = pd.crosstab(cars_csv["Make"],cars_csv['Doors'])
#
# group = cars_csv.groupby("Make")

odometer_col = cars_csv['Odometer (KM)']

cars_csv['Price']=cars_csv['Price'].replace('[\$,]', '', regex=True).astype(float).astype(int)
cars_csv['Make']=cars_csv['Make'].str.lower()
print(cars_csv)


miss_car_sales = pd.read_csv('C:\\Users\\zvallarino\\OneDrive - The Population Council, Inc\\Documents\\DataForPanda\\car-sales-missing-data.csv')
print(miss_car_sales)

miss_car_sales['Odometer'].fillna(miss_car_sales['Odometer'].mean(),inplace=True)

print(miss_car_sales)

miss_car_sales.dropna(inplace=True)

print(miss_car_sales)

miss_car_sales["Seats"] = 5
print(miss_car_sales)

engine_size = pd.Series([1.5,2.2,4.5,3.3,3.1,2.9])

miss_car_sales['Engine Size'] = engine_size

miss_car_sales['Price']=miss_car_sales['Price'].replace('[\$,]', '', regex=True).astype(float).astype(int)


miss_car_sales['Price Per Kilometer'] = miss_car_sales['Odometer']/miss_car_sales['Price']

miss_car_sales.drop('Price Per Kilometer',axis=1 , inplace= True)

shuffled = miss_car_sales.sample(frac=1)

shuffled = shuffled.sort_index()


shuffled['Odometer']=shuffled['Odometer'].apply(lambda x: x/1.6)

shuffled.rename(columns={'Odometer': 'Odometer in Miles'}, inplace=True)

print(shuffled)


