def explore_data(dataframe):
    """
    Explore the given DataFrame and provide an overview of its structure and basic statistics.

    Parameters:
    dataframe: The DataFrame to be explored.

    Returns:
    None
    """

    print("************** EXPLORING DATA **************", "\n")

    # 1. Shape of the dataframe
    print("Number of rows: ", dataframe.shape[0])
    print("Number of columns: ", dataframe.shape[1])

    # 2. Top 5 and Bottom 5 data instances
    print("\n\nData View: First 5 Instances:\n\n")
    print(dataframe.head())
    print("\n\nData View: Last 5 Instances:\n\n")
    print(dataframe.tail())

    # 3. Dataset numeric columns description
    print("\n\nData numeric columns description:\n\n")
    print(dataframe.describe())

    # 4. Dataset information
    print("\n\nData information:\n\n")
    dataframe.info()

    # 5. Dataset Columns
    print("\n\nData columns:\n")
    print(dataframe.columns)

    return dataframe
