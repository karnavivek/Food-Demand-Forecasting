'''
Proof of concept using streamlit
'''

import pandas as pd
import streamlit as st
from forecaster import TSForecaster
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#header
st.title("ONE CLICK FORECASTER")
st.write("This forecaster will help you run your data through Time Series Model")

#user_interctive area (Inputting Data and choose Time Series Forecasting Model of choice - ARIMA, LSTM)
data = st.file_uploader("Upload your Time Series Data here! //must be csv ;)", type='csv')

if data is not None:
    df = pd.read_csv(data)
    st.dataframe(df)

choice = st.selectbox("Select the model of Choice: ", ("ARIMA",'LSTM'))
st.write(f'{choice} model is READY to go! [Check following options to Generate Final Forecast]')

if choice == 'LSTM':
    st.write("| HYPERPARAMTER TUNING SLIDERS |")
    learning_rate = st.slider('Learning Rate:', 0, 10, 0)
    st.write(f"{learning_rate}")
    epochs = st.slider("No. of Epochs:", 1, 300, 0)

elif choice == 'ARIMA':

    st.divider()
    clicked_st = st.button("Check Stationary")
    model = TSForecaster('arima')
    check_stationarity  = model.stationarity_test(df)
    if clicked_st:
        st.write(check_stationarity)

    st.divider()
    lags = st.number_input("Lags",1,30)
    clicked_plot = st.button("Plot PACF & ACF Graphs")
    if clicked_plot:
        plot1, plot2 = model.plot_acf_pacf(df, lags)
        plot1, plot2

    st.divider()
    clicked_make_sta = st.button("Make Stationary")
    new_fod_data  = model.make_stationary(df)
    if clicked_make_sta:
        st.markdown("This is NEW data based on First-Order Differencing:")
        st.dataframe(new_fod_data)
    
    train_data = new_fod_data[:130]
    test_data = new_fod_data[130:]

    def plot_ts(train_data, test_data):
        fig, ax = plt.subplots()
        plt.figure(figsize=(15,5))
        # plt.plot(df_weekly_orders['num_orders'], marker='o')
        ax.plot(train_data, marker='o', label='Training Data')
        ax.plot(test_data, marker='o', label='Testing Data')
        ax.set_xlabel("Weeks")
        ax.legend()
        return st.pyplot(fig)
    
    st.divider()
    clicked_make_plot = st.button("Generate Plot")
    if clicked_make_plot:
        plot_ts(train_data, test_data)
        

    st.divider()


    
    

clicked_generate_forecast = st.button("Generate Forecast")
if clicked_generate_forecast:
    if choice == 'ARIMA':
        # model = TSForecaster('arima')
        # check_stationarity  = model.stationarity_test(df)
        # st.write(check_stationarity)
        pass


    elif choice == 'LSTM':
        '''
        We have to pickle the model & then upload that into streamlit so that data
        can be read on the spot

        problem1: It is not able to read the input data.
        '''
        st.write("Hy")
        # X_train, y_train, X_test, y_test = train_test_split(data, random_state=1)
        # df = pd.read_csv(data)
        # train_data = df[:115]
        # test_data = df[115:]
        # model = TSForecaster('lstm', train_data, test_data)
        # pred = model.run_models('lstm', train_data, test_data)
        # st.write(pred)
        
