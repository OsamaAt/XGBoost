#1 Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import xgboost as xgb 
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error ,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#2 Uploading The Data
data=fetch_california_housing()
x_input,y_output=data.data , data.target
data_frame=pd.DataFrame(x_input , columns=data.feature_names)

#3 Printing 
print(f'Data Shape : {data_frame.shape}')
print(f'Feature Names : {data.feature_names}')
print(f'Print First 5 Rows : {data_frame.head(5)}')

#4 Training And Testing The Data
X_train , X_test , Y_train , Y_test =train_test_split(x_input , y_output , test_size=0.2 , random_state=42)

#5 Preparation 
scaler=StandardScaler()
X_train_Scaler=scaler.fit_transform(X_train)
X_test_Scaler=scaler.transform(X_test)

#6 Training The Model
xgb_model=xgb.XGBRegressor(n_estimators=200 , learning_rate=0.1 , max_depth=4 , random_state=42)
xgb_model.fit(X_train_Scaler , Y_train)

#7 Performance Evaluation
train_score=xgb_model.score(X_train_Scaler , Y_train)
test_score=xgb_model.score(X_test_Scaler , Y_test)
print(f'Training Score : {train_score}')
print(f'Testing Score : {test_score}')

y_pred=xgb_model.predict(X_test)

#8 Comparision Between Actual And Predicted
Compraision=pd.DataFrame({'Actual' : Y_test[:10] , 'Predicted' : y_pred[:10]})
print(f'Prediction VS Actual {Compraision}')

#9  Compute The Mean Error
mae=mean_absolute_error(y_pred , Y_test)
mse=mean_squared_error(y_pred , Y_test)
rmse=np.sqrt(mse)
print(f'Mean Absolute Error : {mae:.2f}')
print(f'Mean Seqeared Error : {mse:.2f}')
print(f'Root Mean Sequred Error : {rmse:.2f}')

#10 Plotting The Features
importance=xgb_model.feature_importances_

plt.barh(data.feature_names , importance ,color='skyblue')
plt.title('XGBoost Features Importance (fetch_california_housing)')
plt.xlabel('Feature Importance Score')
plt.show()
