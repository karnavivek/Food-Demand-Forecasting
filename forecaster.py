import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, LeakyReLU, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam

class TSForecaster:

    def __init__(self, model_type: str, train_data=None, test_data=None):
        self.model_type = model_type
        self.train_data = train_data
        self.test_data = test_data

    def plot_time_series(self, data):
        pass #do it later

    def plot_acf_pacf(self, data, lag):
        pacf = plot_pacf(data, lags=lag)
        acf = plot_acf(data, lags=lag)

    def stationarity_test(self, data):
        '''
        Here we will perform the Augmented-Dickey Fuller test to check stationarity of the data
        '''
        val_order_ = adfuller(data, autolag='AIC')
        print(f'T-stat: {val_order_[0]}')
        print(f'P Value: {val_order_[1]}')
        print(f'Lags: {val_order_[2]}')
        print(f'Number of Observations used: {val_order_[3]}')
        for key, val in val_order_[4].items():
            print(f'Critical Value at {key}: {val}\n')

        if val_order_[1]>0.05:
            return 'This Time Series Model is NON STATIONARY, may give bad predictions! :('
        else:
            return 'This Time Series Model is STATIONARY, getting prediction is recommended! :)'

    def make_stationary(self, data, order=1):
        '''
        To make the TS data stationary, we are going to perform first order differencing
        '''
        return data.diff(periods=order).dropna().reset_index(drop=True)
        

    def create_dataset(self, data, window_size=5):
        X_data, y_data = [], []

        for i in range(window_size, len(data)):
            X_data.append(data[i-window_size:i, 0])
            y_data.append(data[i,0])

        return np.array(X_data), np.array(y_data)
    
    def run_models(self, model_type: str, train_data, test_data, window_size=5):
        '''
        step-1 = check the stationarity of the train_data
        step-2 = check if the TS data is stationary? if it is -> we will go to step 4, if not -> go to step-3
        step-3 = 

        '''
        if model_type == 'arima':
            #scale the data using MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_train_data = scaler.fit_transform(train_data)
            scaled_test_data = scaler.fit_transform(test_data)


        elif model_type == 'lstm':
            #scale the data using MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_train_data = scaler.fit_transform(train_data)
            scaled_test_data = scaler.fit_transform(test_data)

            #we need to create rolling data for lstm for it to be able to read it properly

            window_size = 5
            X_train, y_train = self.create_dataset(scaled_train_data, window_size)
            # X_val, y_val = create_dataset(val_orders, window_size)
            X_test, y_test = self.create_dataset(scaled_test_data, window_size)

            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            # X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

            '''Creating LSTM Model using tensorflow.keras'''

            model = Sequential()
            model.add(LSTM(128, input_shape = (1, window_size), return_sequences=True))
            model.add(LeakyReLU(alpha=0.5))
            model.add(LSTM(128, return_sequences=True))
            model.add(LeakyReLU(alpha=0.5))
            model.add(Dropout(0.3))
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(0.3))
            model.add(Dense(1, 'linear'))

            cp = ModelCheckpoint('models_cp/model.keras', save_best_only = True)

            # early_stopping = EarlyStopping(monitor = 'val_loss', 
            #                                patience = 2, 
            #                                mode = 'min')

            model.compile(loss=MeanSquaredError(), 
                        optimizer=Adam(learning_rate=0.0001),
                        metrics=[MeanAbsoluteError()])

            model.fit(X_train, y_train, shuffle = False, epochs=100, callbacks=[cp])

            #Predicted data of the above model
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_pred = np.reshape(train_pred, (train_pred.shape[0], train_pred.shape[2]))
            # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[2]))

            # val_pred = np.reshape(val_pred, (val_pred.shape[0], val_pred.shape[2]))
            # X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[2]))

            test_pred = np.reshape(test_pred, (test_pred.shape[0], test_pred.shape[2]))
            # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[2]))

            train_pred = scaler.inverse_transform(train_pred.reshape(-1,1))
            y_train = scaler.inverse_transform(y_train.reshape(-1,1))

            # val_pred = scaler.inverse_transform(val_pred.reshape(-1,1))
            # y_val = scaler.inverse_transform(y_val.reshape(-1,1))

            test_pred = scaler.inverse_transform(test_pred.reshape(-1,1))
            y_test = scaler.inverse_transform(y_test.reshape(-1,1))

            return test_pred

        elif model_type == 'arma':
            pass
