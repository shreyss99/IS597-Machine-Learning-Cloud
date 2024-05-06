import pandas as pd


def load_data(filename):
    """
    Read in input file and load data

    Parameters:
    filename: csv file

    Returns:
    dataframe
    """

    print("************** LOADING DATA **************", "\n")
    data = pd.read_csv(open(filename, "rb"), encoding="utf-8")
    print("Dataset has been loaded successfully", "\n")
    return data
