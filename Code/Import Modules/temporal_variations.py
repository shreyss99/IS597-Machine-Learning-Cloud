from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


def create_decomposition(dataframe, feature):
    """
    Decompose the time series data into its components: trend, seasonality, and residuals.

    Parameters:
    dataframe: The DataFrame containing the time series data.
    feature: The name of the column representing the time series feature.

    Returns:
    tuple: A tuple containing arrays representing the trend, seasonality, and residuals.
    """

    print("************** TEMPORAL VARIATIONS **************", "\n")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe[feature].values.reshape(-1, 1))

    decomposition = seasonal_decompose(scaled_data, model='additive', period=365)

    # Plot decomposed components
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    print("************** VISUALIZING DECOMPOSITION COMPONENTS **************", "\n")

    # Plot the decomposed components
    plt.figure(figsize=(10, 8))
    plt.subplot(411)
    plt.plot(scaled_data, label='Original', color='lightblue')
    plt.legend(loc='best')

    plt.subplot(412)
    plt.plot(trend, label='Trend', color='red')
    plt.legend(loc='best')

    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality', color='lightgreen')
    plt.legend(loc='best')

    plt.subplot(414)
    plt.plot(residual, label='Residuals', color='orange')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()
