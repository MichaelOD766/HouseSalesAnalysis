import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression


#Load the CSV
filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv"
df = pd.read_csv(filename)

df.head()

#Display the data types of each column using the function dtypes
print(df.dtypes)

#Statistical summary of the dataframe
df.describe()

#Drop the columns "id" and "unnamed: 0" from axis 1 and then use describe method
df.drop(['id', 'Unnamed: 0'], axis = 1 , inplace = True)
df.describe()

#We can see that we have missing values for the bedrooms and bathrooms columns
print("Number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("Number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

#Replace the missing values
mean = df['bedrooms'].mean()
df['bedrooms'].replace(np.nan, mean, inplace = True)

mean = df['bathrooms'].mean()
df['bathrooms'].replace(np.nan, mean, inplace = True)

print("Number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("Number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

#Use 'value_counts' method to count the number of house with unique floor values
df['floors'].value_counts().to_frame()

#Use boxplot function to determine outliers on waterfront houses
sns.boxplot(x = 'waterfront', y = 'price', data = df)

#Use regplot to determine if the feature 'sqft_above' is neg or pos correlated with price
sns.regplot(x = 'sqft_above', y = 'price', data = df)

#Use the 'corr()' method to find the feature most correlated with price
df.corr()['price'].sort_values()

#Linear regression model using 'long' and calculate R^2
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X,Y)

#Linear regression model to predict price using 'sqft_living' and calculate R^2
X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X,Y)

#Fit a linear regression model to predict price using a list of features, calculate R^2
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]
X = df[features]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X,Y)

#Tuples with first element as estimator and second element model constructor
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

#Use tuples to create a pipeline object to predict the 'price' and fit the features in the list, then calculate R^2
pipe = Pipeline(Input)
pipe
pipe.fit(X,Y)
pipe.score(X,Y)



from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#Split the data into training and test sets
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

#Create and fit a Ridge regression object using the training data
from sklearn.linear_model import Ridge
RidgeModel = Ridge(alpha = 0.1)
RidgeModel.fit(x_train, y_train)
RidgeModel.score(x_test, y_test)

#Second order polynomial transform on both the training data and testing data
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
poly = Ridge(alpha=0.1)
poly.fit(x_train_pr, y_train)
poly.score(x_test_pr, y_test)


