import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_data(dataframe):
    """
    Visualize the data in the given DataFrame using various plots and charts.

    Parameters:
    dataframe: The DataFrame containing the data to be visualized.

    Returns:
    None
    """

    print("************** VISUALIZING DATA **************", "\n")

    # A) Distribution of Speed Violations
    print("A) Distribution of Speed Violations", "\n")
    plt.figure(figsize=(10, 6))
    sns.histplot(dataframe['VIOLATIONS'], bins=50, kde=True)
    plt.xlabel('Speed Violations')
    plt.ylabel('Frequency')
    plt.title('Distribution of Speed Violations')
    plt.show()

    # B) Distribution of Speed Violations by Year
    print("B) Distribution of Speed Violations by Year", "\n")
    dataframe['YEAR'] = dataframe['VIOLATION_DT'].dt.year
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='YEAR', y='VIOLATIONS', data=dataframe)
    plt.xlabel('Year')
    plt.ylabel('Speed Violations')
    plt.title('Distribution of Speed Violations by Year')
    plt.show()

    # C) Distribution of Total Speed Violations grouped by Year
    print("C) Distribution of Total Speed Violations grouped by Year", "\n")
    violations_per_year = dataframe.groupby('YEAR')['VIOLATIONS'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='YEAR', y='VIOLATIONS', data=violations_per_year)
    plt.xlabel('Year')
    plt.ylabel('Total Speed Violations')
    plt.title('Total Speed Violations by Year')
    plt.show()

    # D) Distribution of Speed Violations by Month
    print("D) Distribution of Speed Violations by Month", "\n")
    dataframe['MONTH'] = dataframe['VIOLATION_DT'].dt.month
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='MONTH', y='VIOLATIONS', data=dataframe)
    plt.xlabel('Month')
    plt.ylabel('Speed Violations')
    plt.title('Distribution of Speed Violations by Month')
    plt.show()

    # E) Chicago Speed Camera Violations by Address
    print("E) Distribution of Total Speed Violations", "\n")
    violation_list = []
    unique_addresses = dataframe.ADDRESS.unique()

    for address in unique_addresses:
        temp_df = dataframe.loc[
            dataframe['ADDRESS'] == address, ['VIOLATION_DT', 'VIOLATIONS']]
        temp_df['VIOLATION_DT'] = pd.to_datetime(temp_df['VIOLATION_DT'])
        temp_df.set_index('VIOLATION_DT', inplace=True)
        temp_df = temp_df.resample('D').sum()
        violation_list.append(temp_df)

    plt.figure(figsize=(12, 6), dpi=100, facecolor="w")
    for address, data in zip(unique_addresses, violation_list):
        plt.plot(data.index, data['VIOLATIONS'], label=address)

    plt.ylabel("Violations")
    plt.xlabel("Date")
    plt.title("Chicago Speed Camera Violations by Address")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=4)
    plt.show()

    # F) Trend and Seasonality of Chicago Speed Violations
    print("F) Trend and Seasonality of Chicago Speed Violations", "\n")
    plt.figure(figsize=(12, 6))
    dataframe['VIOLATIONS'].plot(label='Speed Violations', alpha=0.5)
    dataframe['VIOLATIONS'].rolling(window=30).mean().plot(label='30-Day Rolling Mean', alpha=0.7)
    dataframe['VIOLATIONS'].rolling(window=30).std().plot(label='30-Day Rolling Std', alpha=0.3)
    plt.xlabel('VIOLATION_DT')
    plt.ylabel('Speed Violations')
    plt.title('Trend and Seasonality of Chicago Speed Violations')
    plt.legend()
    plt.show()
