from statsmodels.tsa.arima.model import ARIMA


def train_arima_model(dataframe, endogenous_feature, p_value, d_value, q_value):
    """
    Train an ARIMA model on the given DataFrame.

    Parameters:
    dataframe: The DataFrame containing the data.
    endogenous_feature: The name of the endogenous feature to be used as the target variable.
    p_value (int): The autoregressive order (p) of the ARIMA model.
    d_value (int): The differencing order (d) of the ARIMA model.
    q_value (int): The moving average order (q) of the ARIMA model.

    Returns:
    ARIMAResultsWrapper: Trained ARIMA model results.
    """

    print("************** TRAINING ARIMA MODEL **************", "\n")

    endogenous = dataframe[endogenous_feature]
    arima_model = ARIMA(endogenous, order=(p_value, d_value, q_value))
    arima_fit = arima_model.fit()

    print("************** ARIMA MODEL SUMMARY **************", "\n")
    print(arima_fit.summary())

    return arima_fit
