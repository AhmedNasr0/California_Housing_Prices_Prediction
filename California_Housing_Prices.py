import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import data
data=pd.read_csv('housing.csv')
draw=plt.scatter(
    x=data['longitude'],
    y=data['latitude'],
    alpha=0.3,
    s=data['population']/100,
    label='Population',
    c=data['median_house_value'],
    cmap=plt.get_cmap("jet")
)
plt.colorbar(label='Median House Value')
import matplotlib.image as mpimg
californiaImg=mpimg.imread("map_of_California.png")
plt.imshow(californiaImg,extent=[-124.8,-113.8,32.45,42],alpha=0.4)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.legend()
plt.show()

# data cleaning

# handle total_bedrooms null values with the mean of the column
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
data['total_bedrooms']=imputer.fit_transform(data[['total_bedrooms']])

# data preprocessing
# handle the categorical values
from sklearn.preprocessing import OneHotEncoder

cols=data.shape[1]
y=data.iloc[:,cols-1:].values
x=data.iloc[:,:cols-1].values

onehotencoder=OneHotEncoder()
ohe=onehotencoder.fit_transform(x[:,x.shape[1]-1].reshape(-1,1)).toarray()
x=np.delete(x,x.shape[1]-1,axis=1)
x=np.concatenate([ohe,x],axis=1)

# handle dummy variables trap
x=x[:,1:x.shape[1]]

# split data to train and set DataSet
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 , random_state=0)


# choose the features
import statsmodels.api as sm
x=np.append(arr=np.ones((x.shape[0],1)).astype(int),values=x,axis=1)
x_optimal=x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
regressor_OLS=sm.OLS(y.tolist(),x_optimal.tolist()).fit()
# print(regressor_OLS.summary())

# Model selection and evaluation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
# regressor=LinearRegression()
# regressor.fit(x_train,y_train)
# y_prediction=regressor.predict(x_test)
# print("multi :",mean_absolute_error(y_test,y_prediction))
#
#
# from sklearn.svm import SVR
# svr_regressor=SVR(kernel='rbf')
# svr_regressor.fit(x_train,y_train.ravel())
# svr_y_prediction=svr_regressor.predict(x_test)
# print("SVR : " , mean_absolute_error(y_test,svr_y_prediction))
#
# from sklearn.tree import DecisionTreeRegressor
# tree_regressor=DecisionTreeRegressor(random_state=0)
# tree_regressor.fit(x_train,y_train)
# tree_y_prediction=tree_regressor.predict(x_test)
# print("Tree : " , mean_absolute_error(y_test,tree_y_prediction))
#
# from sklearn.ensemble import RandomForestRegressor
# forest_regressor=RandomForestRegressor(n_estimators=500,random_state=0)
# forest_regressor.fit(x_train,y_train.ravel())
# forest_y_prediction=forest_regressor.predict(x_test)
# print("Forest : " , mean_absolute_error(y_test,forest_y_prediction))
#
import xgboost as xgb
xgb_regressor = xgb.XGBRegressor(n_estimators=300,learning_rate=0.2,max_depth=5)
xgb_regressor.fit(x_train, y_train)
xgb_y_prediction = xgb_regressor.predict(x_test)
# print("XGB : " , mean_absolute_error(y_test,xgb_y_prediction))

#  visualize the models
mae_values = {
    'Multiple Linear Regression': 49697.070164811674,
    'SVR': 87481.4907685241,
    'Decision Tree': 43848.08042635659,
    'Random Forest': 31278.153456879845,
    'XGBoost': 29696.10901257419
}

models = list(mae_values.keys())
mae = list(mae_values.values())

plt.bar(models, mae, color='red')
plt.xlabel('Mean Absolute Error (MAE)')
plt.title('MAE Comparison of Different Models')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, xgb_y_prediction, label='Predicted vs Actual')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='red')
plt.xlabel('Actual Values')
plt.ylabel("Predicted Values")
plt.show()

