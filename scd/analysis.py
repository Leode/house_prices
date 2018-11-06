import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv('../data/train.csv')

#remove non-numerical data
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
train_data = train_data.select_dtypes(include=numerics)
print(train_data.shape)

#replace missing with zeros
train_data.fillna(0, inplace = True)

#Split data
independent_variables = list(train_data.columns.values)
independent_variables.remove('SalePrice')

X_train = train_data[independent_variables].values
y_train = train_data[['SalePrice']].values.ravel()



#benchmark classifier
regr = RandomForestRegressor(n_estimators=100, criterion = "mse", max_depth=2)
regr.fit(X_train, y_train)

#Predict on test data
y_pred = regr.predict(X_train)
rms = np.sqrt(mean_squared_error(y_train, y_pred))
print (rms)

