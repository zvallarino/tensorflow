import pandas as pd
import matplotlib.pyplot as plt
# 2 main datatypes

series = pd.Series(['Gundam', "Anime", "god"])

# print(series)

# note series are one dimesional

colors = pd.Series(['red','blue', 'green'])
#
# print(colors)

car_data =pd.DataFrame({"car make":series, 'colors':colors})
## takes a python list, you can make it using Series
# print(car_data)

car_sales = pd.read_csv('C:\\Users\\zvallarino\\OneDrive - The Population Council, Inc\\Documents\\DataForPanda\\car-sales.csv')
#
#
# # print(car_sales)
#
# # car_sales.to_csv("C:/Users/zvallarino/OneDrive - The Population Council, Inc/Documents/DataForPanda/export_car_sales.csv", index=False)
#
#
#
# exported_car_sales = pd.read_csv("C:/Users/zvallarino/OneDrive - The Population Council, Inc/Documents/DataForPanda/export_car_sales.csv")
# # print(exported_car_sales)
# # print(car_sales.dtypes)
# # print(car_sales.columns)
# #
# # print(car_sales.describe())
#
# ## viewing and selecting data
#
# print(car_sales.head())
#
# ## it gives you the first 5 rows
#
# print(car_sales.head(7))

##it gives you the first 7 rows

# print(car_sales.tail())
#
# ## the last 5 rows
#
# animals = pd.Series(['cat','dog','bird','panda','eagle'], index=[1,8,6,3,10])
#
# print(animals)
# ##loc refers to index
# print(animals.loc[3])
# ## iloc is position
# print(animals.iloc[1])
# print(animals.loc[:6])
#
# ## conditional statements like I only want toyotas or i only want odometers
# print(car_sales[car_sales['Make']=="Toyota"])
# print(car_sales[car_sales['Odometer (KM)'] >= 100000])

## compares two columns
# print(pd.crosstab(car_sales['Make'], car_sales["Doors"]))
#
# print(car_sales)
##Groupby
## it groups the data by make and gives you the mean
# car_sales.groupby(['Make']).mean()
# car_sales[car_sales['Make']=="Toyota"].groupby().mean()



# car_sales['Price'] = car_sales['Price'].replace('[\$,]', '', regex=True).astype(float).astype(int)
# car_sales["Price"].plot()
# plt.show()
# print(car_sales['Make'].str.lower())
#
# car_sales['Make'] = car_sales['Make'].str.lower()
#
# print(car_sales)

miss_car_sales = pd.read_csv('C:\\Users\\zvallarino\\OneDrive - The Population Council, Inc\\Documents\\DataForPanda\\car-sales-missing-data.csv')

# print(miss_car_sales)

miss_car_sales["Odometer"].fillna(miss_car_sales['Odometer'].mean(), inplace=True)
print(miss_car_sales)

miss_car_sales.dropna(inplace=True)

print(miss_car_sales)

## Creating data from existing

print(miss_car_sales)

seats_column = pd.Series([5, 5, 5, 5, 5])

car_sales["Seat"] = seats_column

print(car_sales)

# car_sales['Seat'].fillna(4,inplace=True)

print(car_sales)


fuel_economy = [7.5,8.0,5.0,9.6,8.7, 5.3, 6.1, 9.9, 2.3, 5.3]

car_sales['Fuel per 60 miles'] = fuel_economy
car_sales['Total Fuel Used'] = car_sales["Odometer (KM)"]/100 * car_sales['Fuel per 60 miles']

print(car_sales)

# car_sales["Wheels"] = 4

print(car_sales)

car_sales.drop('Total Fuel Used',axis=1, inplace=True)

print(car_sales)



car_sales_shuffled = car_sales.sample(frac=1)

print(car_sales)
print(car_sales_shuffled
      )
##half the date
# car_sales.sample(frac=.5)

