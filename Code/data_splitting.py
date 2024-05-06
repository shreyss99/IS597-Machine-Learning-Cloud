def split_data(dataframe):
    """
    Split the given DataFrame into training and testing sets.

    Parameters:
    dataframe: The DataFrame containing the data to be split.

    Returns:
    tuple: A tuple containing the training and testing datasets.
    """

    print("************** SPLITTING DATA **************", "\n")

    # Since we do not have a label column we cannot use the train_test_split method for data split
    train_data = dataframe.iloc[:int(0.8 * len(dataframe))]
    test_data = dataframe.iloc[int(0.8 * len(dataframe)):]

    # TSizes of Train and Test Data
    print("\nNumber of rows in training dataset: ", train_data.shape[0])
    print("\nNumber of columns in training dataset: ", train_data.shape[1])

    print("\nNumber of rows in testing dataset: ", test_data.shape[0])
    print("\nNumber of columns in testing dataset: ", test_data.shape[1])

    return train_data, test_data
