import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from scipy import stats

def plot_predictions(test_timestamps, y_test, y_pred, model_name, save_path):
    '''
    ! this function will plot the test dataset plus its forecast by model.
    '''

    plt.figure(figsize=(12, 8))
    plt.plot(test_timestamps, y_test, '-', label="test data", color="midnightblue")
    plt.plot(test_timestamps, y_pred, '-.', label=f"{model_name} model", color="crimson")
    plt.legend()
    plt.xlabel('timestamp')
    plt.ylabel('temperature (°C)')
    plt.title(f"forecasting for {model_name} model")
    plt.savefig(save_path)
    plt.close()


def overfitting_test(best_model, y_train, y_train_pred, mse_mean):
    '''
    ! this function will evaluate the overfitting of a model using cross-validation.

    ! best_model: current trained model
    ! y_train: input features for training
    ! y_train_pred: forecasting of the trained data
    ! mse_mean: mean squared error on the test dataset
    ! score: a measurement of overfitting
    '''

    mse_train = -cross_val_score(best_model, y_train, y_train_pred.ravel(), cv=5, scoring='neg_mean_squared_error')
    score = np.abs((mse_mean - mse_train.mean())/mse_train.mean())

    return score

directory = os.path.dirname(os.path.abspath('__file__'))
os.chdir(directory)
filename = "southeast_daily_mean" 
csv_path = os.path.join(directory, filename)
df = pd.read_parquet(filename) # will read faster

timeseries = df.set_index('date_time')

train_start_dt = '2000-05-07'
test_start_dt = '2021-01-01'

train = timeseries.copy()[(timeseries.index >= train_start_dt) & (timeseries.index < test_start_dt)][['temp']]
test = timeseries.copy()[timeseries.index >= test_start_dt][['temp']]

power = PowerTransformer(method='box-cox', standardize=True)
train_data = power.fit_transform(train)
test_data = power.transform(test)

timesteps = 2
train_data_timesteps = np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
test_data_timesteps = np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]

regressors = [
    (Lasso(), {'alpha': [0.1, 1, 10]}),
    (Ridge(), {'alpha': [0.1, 1, 10]}),
    (DecisionTreeRegressor(), {'max_depth': [None, 10, 20, 30, 40, 100]}), 
    (RandomForestRegressor(), {'n_estimators': [50, 100, 150, 200]}),
    (GradientBoostingRegressor(), {'n_estimators': [100], 'learning_rate': [0.01, 0.1, 0.5], 'max_depth': [3, 4, 5], 'subsample': [0.8]}),
    (SVR(), {'kernel': ['rbf'], 'C': [50, 100, 150], 'gamma': ['auto'], 'epsilon': [0.01, 0.1, 0.5]})
]

results = []

for regressor, param_grid in regressors:
    print(regressor)

    x_train, y_train = train_data_timesteps[:,:timesteps-1], train_data_timesteps[:,[timesteps-1]]
    x_test, y_test = test_data_timesteps[:,:timesteps-1], test_data_timesteps[:,[timesteps-1]]

    grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train,  y_train.ravel())

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    y_train_pred = best_model.predict(x_train)
    y_test_pred = best_model.predict(x_test)

    y_train_pred = best_model.predict(x_train).reshape(-1,1)
    y_test_pred = best_model.predict(x_test).reshape(-1,1)

    y_train = power.inverse_transform(y_train)
    y_train_pred = power.inverse_transform(y_train_pred)
    y_test = power.inverse_transform(y_test)
    y_test_pred = power.inverse_transform(y_test_pred)

    mse_scores = -cross_val_score(best_model, y_test, y_test_pred.ravel(), cv=5, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(best_model, y_test, y_test_pred.ravel(), cv=5, scoring='r2')
    mae_scores = -cross_val_score(best_model, y_test, y_test_pred.ravel(), cv=5, scoring='neg_mean_absolute_error')

    mse_mean = mse_scores.mean()
    mse_std = mse_scores.std()
    mse_var = mse_scores.var()
    r2_mean = r2_scores.mean()
    mae_mean = mae_scores.mean()

    overfitting_score = overfitting_test(best_model, y_train, y_train_pred, mse_mean)

    results.append({
        'regressor': regressor.__class__.__name__,
        'overfitting?': overfitting_score,
        'best_params': best_params,
        'mse_mean': mse_mean,
        'mse_std': mse_std,
        'mse_variance': mse_var,
        'r2_mean': r2_mean,
        'mae_mean': mae_mean,
    })

    train_timestamps = timeseries[(timeseries.index < test_start_dt) & (timeseries.index >= train_start_dt)].index[timesteps-1:]
    test_timestamps = timeseries[test_start_dt:].index[timesteps-1:]

    model_name = regressor.__class__.__name__
    save_path = f"plots/{model_name}_.png"
    plot_predictions(test_timestamps, y_test, y_test_pred, model_name, save_path)

for result in results:
    print(result)

results_df = pd.DataFrame(results)
results_df.to_csv('regressor_results.csv', index=False)



