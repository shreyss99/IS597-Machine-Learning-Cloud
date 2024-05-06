import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def arima_model_evaluation(dataframe, endogenous, arima_model):
    """
    Evaluate the ARIMA model's performance.

    Parameters:
    dataframe: The DataFrame containing the data.
    endogenous: The name of the endogenous variable in the ARIMA model.
    arima_model: The trained ARIMA model.

    Returns:
    dict: A dictionary containing evaluation metrics (MSE, RMSE, MAE).
    """

    forecast_result = arima_model.get_forecast(steps=len(dataframe[endogenous]))
    forecasted_violations = forecast_result.predicted_mean

    # Calculating MSE
    mse = mean_squared_error(dataframe[endogenous], forecasted_violations)

    # Calculating RMSE
    rmse = np.sqrt(mse)

    # Calculating MAE
    mae = mean_absolute_error(dataframe[endogenous], forecasted_violations)

    print("************** ARIMA MODEL EVALUATION **************", "\n")

    print(f"Mean Squared Error (MSE): {mse}", "\n")
    print(f"Root Mean Squared Error (RMSE): {rmse}", "\n")
    print(f"Mean Absolute Error (MAE): {mae}", "\n")
