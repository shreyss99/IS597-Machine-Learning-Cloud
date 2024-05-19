import pandas as pd
import matplotlib.pyplot as plt


def forecast_arima_model(dataframe, endogenous_feature, arima_model, address_name):
    """
    Forecast using an ARIMA model for a specific address.

    Parameters:
    dataframe: The DataFrame containing the data.
    endogenous_feature: The name of the endogenous feature used in the ARIMA model.
    arima_model: The trained ARIMA model.
    address_name: The name of the address for which to forecast the violations.

    Returns:
    float: The forecasted number of violations for the specified address.
    """

    print("************** FORECASTING ARIMA MODEL **************", "\n")

    # Select specific address
    specific_address_data = dataframe[dataframe['ADDRESS'] == address_name]

    # Prepare data
    specific_address_data.loc[:, 'VIOLATION_DT'] = pd.to_datetime(specific_address_data['VIOLATION_DT'])
    specific_address_data.set_index('VIOLATION_DT', inplace=True)
    specific_address_data.sort_index(inplace=True)

    # Daily frequency and impute missing values with 0
    specific_address_daily = specific_address_data.resample('D').sum().fillna(0)

    # If there is enough data for forecasting
    if len(specific_address_daily) < 365:
        print("Insufficient data for forecasting. Please ensure you have at least 365 days of data.")
    else:
        model = arima_model

        # Forecast
        forecast = model.forecast(steps=365)  # Forecasting for the next year

        # Plot forecast
        plt.plot(specific_address_daily.index, specific_address_daily[endogenous_feature], label='Actual',
                 color='orange')
        plt.plot(specific_address_daily.index[-1] + pd.to_timedelta(range(1, 366), unit='D'), forecast,
                 label='Forecast',
                 color='blue')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Number of Violations')
        plt.title(f'ARIMA Forecast for Speed Violations For Next 1 Year for Address {address_name}')
        plt.show()
