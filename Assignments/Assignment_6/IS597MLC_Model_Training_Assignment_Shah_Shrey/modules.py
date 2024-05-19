#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[2]:


def load_data(filename, colname):
    """
    Read in input file and load data

    filename: csv file
    colname: column name for texts
    return: X and y dataframe
    """

    ## 1. Read in data from input file
    df = pd.read_csv(filename, sep="\t", encoding='utf-8')
    
    print("************** Loading Data ************", "\n")

    # Check number of rows and columns
    print("No of Rows: {}".format(df.shape[0]))
    print("No of Columns: {}".format(df.shape[1]))

    ## 2. Select data needed for processing
    print(f"Selecting columns needed for processing: pmid, {colname}, rct", "\n")
    df = df[['pmid', colname, 'rct']]
    

    ## 3. Cleaning data
    # Trim unnecessary spaces for strings
    df[colname] = df[colname].apply(lambda x: str(x))

    # 3-1. Remove null values
    df=df.dropna()

    # Check number of rows and columns
    print("No of rows (After dropping null): {}".format(df.shape[0]))
    print("No of columns: {}".format(df.shape[1]))

    # 3-2. Remove duplicates and keep first occurrence
    df.drop_duplicates(subset=['pmid'], keep='first', inplace=True)

    # Check number of rows and columns
    print("No of rows (After removing duplicates): {}".format(df.shape[0]))

    # Check the first few instances
    print("\n<Data View: First Few Instances>\n")
    print(df.head(5))
    
    # 3-3. Check label class
    print('\nClass Counts(label, row): Total')
    print(df["rct"].value_counts())
    

    ## 4. Split into X and y (target)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    return X, y


# In[5]:


def split_data(X_data, y_data):
    """
    Read in the X_data and y_data and split into train, validtion, and test sets.

    X_data: dataframe consisting of only the input features
    y_data: series consisting of only the output label
    return: a tuple of the split data consisting of train, test and validation sets for both X_data and y_data
    
    """

    print("\n************** Spliting Data **************\n")

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
    X_val, X_test, y_val, y_test = train_test_split(X_test,y_test, test_size=0.5, random_state=42, stratify=y_test)

    ## Check the data view of each data set

    ## Data Shape
    print("Train Data: {}".format(X_train.shape))
    print("Val Data: {}".format(X_val.shape))
    print("Test Data: {}".format(X_test.shape))

    ## Label Distribution
    print('\nClass Counts(label, row): Train')
    print(y_train.value_counts())
    print('\nClass Counts(label, row): Validation')
    print(y_val.value_counts())
    print('\nClass Counts(label, row): Test')
    print(y_test.value_counts())

    ## Display the first 3 instances of X data
    print("\nData View: X Train")
    print(X_train.head(3))
    print("\nData View: X Val")
    print(X_val.head(3))
    print("\nData View: X Test")
    print(X_test.head(3))

    ## Reset index

    print("\n************** Resetting Index **************\n")

    # Train Data
    X_train=X_train.reset_index(drop=True)
    y_train=y_train.reset_index(drop=True)

    # Validation Data
    X_val=X_val.reset_index(drop=True)
    y_val=y_val.reset_index(drop=True)

    # Test Data
    X_test=X_test.reset_index(drop=True)
    y_test=y_test.reset_index(drop=True)

    ## Check data

    ## Data Shape
    print("Train Data: {}".format(X_train.shape))
    print("Validation Data: {}".format(X_val.shape))
    print("Test Data: {}".format(X_test.shape))

    ## Label Distribution
    print('\nClass Counts(label, row): Train\n')
    print(y_train.value_counts())
    print('\nClass Counts(label, row): Val\n')
    print(y_val.value_counts())
    print('\nClass Counts(label, row): Test\n')
    print(y_test.value_counts())

    ## Display the first 3 instances of X data
    print("\nData View: X Train")
    print(X_train.head(3))
    print("\nData View: X Val")
    print(X_val.head(3))
    print("\nData View: X Test")
    print(X_test.head(3))
    
    return (X_train, X_val, X_test, y_train, y_val, y_test)


# In[6]:


def preprocess_data(X_data_raw):
    """
       Function to preprocess data with lowercase conversion, punctuation removal, tokenization, stemming

       X_data_raw: X data in dataframe
       return: transformed dataframe

    """

    print("\n************** Pre-processed Data **************\n")
    
    X_data=X_data_raw.iloc[:, -1].astype(str)
    print(f"\nTrain Data: {X_data.shape}")
    
    ## 1. convert all characters to lowercase
    X_data = X_data.map(lambda x: x.lower())

    ## 2. remove punctuation
    X_data = X_data.str.replace('[^\w\s]', '')

    ## 3. tokenize sentence
    X_data = X_data.apply(nltk.word_tokenize)

    ## 4. remove stopwords
    stopword_list = stopwords.words("english")
    X_data = X_data.apply(lambda x: [word for word in x if word not in stopword_list])

    ## 5. stemming
    stemmer = PorterStemmer()
    X_data = X_data.apply(lambda x: [stemmer.stem(y) for y in x])

    ## 6. removing unnecessary space
    X_data = X_data.apply(lambda x: " ".join(x))

    # Check data view
    print("\nData View: X Train\n")
    print(X_data.head(3))

    return X_data


# In[ ]:




