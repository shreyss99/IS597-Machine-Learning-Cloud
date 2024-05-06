from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt


def get_optimal_parameters(dataframe, feature):
    """
    Determine the optimal parameters for the ARIMA model.

    Parameters:
    dataframe: The DataFrame containing the data.
    feature: The name of the feature to be used in the ARIMA model.

    Returns:
    tuple: A tuple containing the optimal ARIMA parameters (p, d, q).
    """

    print("************** OPTIMAL PARAMETERS FOR ARIMA MODEL **************", "\n")

    # PACF for 'p'
    print("************** OPTIMAL AUTOREGRESSIVE PARAMETERS 'p' **************", "\n")
    plot_pacf(dataframe[feature].diff().dropna())
    plt.show()

    # ACF for 'd'
    print("************** OPTIMAL DIFFERENCING PARAMETERS 'd' **************", "\n")
    # Plot the ACF for the original data and its differences
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(hspace=0.5)
    plot_acf(dataframe[feature], ax=ax1)
    plot_acf(dataframe[feature].diff().dropna(), ax=ax2)
    plot_acf(dataframe[feature].diff().diff().dropna(), ax=ax3)
    plt.show()

    # ACF for 'q'
    print("************** OPTIMAL MOVING AVERAGES PARAMETERS 'q' **************", "\n")
    plot_acf(dataframe[feature].diff().dropna())
    plt.show()
