import pandas as pd


def preprocess_data(dataframe):
    """
    Preprocess the given DataFrame by performing necessary data cleaning and transformation steps.

    Parameters:
    dataframe: The DataFrame containing the raw data to be preprocessed.

    Returns:
    dataframe: The preprocessed DataFrame.
    """

    print("************** CLEANING DATA **************", "\n")

    # A) Removing duplicate rows
    dataframe.drop_duplicates(inplace=True)
    print("\nNumber of rows after removing duplicates: ", dataframe.shape[0])
    print("Number of columns after removing duplicates:", dataframe.shape[1])

    # B) Converting the 'Date' column from object to datetime format and creating a new column VIOLATION_DATE
    dataframe["VIOLATION_DT"] = pd.to_datetime(dataframe["VIOLATION DATE"])
    dataframe.set_index('VIOLATION_DT', inplace=True)
    dataframe = dataframe.reset_index()
    print("\n\nDataset information after date conversion:\n")
    dataframe.info()

    # C) Impute 0 for location fields which are empty
    columns_to_replace = ['LONGITUDE', 'LATITUDE', 'X COORDINATE', 'Y COORDINATE', 'LOCATION']
    dataframe[columns_to_replace] = dataframe[columns_to_replace].fillna(0)

    # D) Impute 0 for 'VIOLATIONS' where it is empty
    dataframe['VIOLATIONS'] = dataframe['VIOLATIONS'].fillna(0)
    print("\n\nDataset after imputation:\n")
    print(dataframe.head())

    # E) Additional exploration
    unique_addresses = dataframe.ADDRESS.unique()
    number_of_addresses = len(unique_addresses)
    print("\n\nUnique addresses count: {}".format(number_of_addresses))

    idx = pd.date_range(dataframe.VIOLATION_DT.min(), dataframe.VIOLATION_DT.max())

    print("\n\nMinimum violation date is {}, Maximum violation date is {}\n\n".format(dataframe.VIOLATION_DT.min(),
                                                                                      dataframe.VIOLATION_DT.max()))

    return dataframe
