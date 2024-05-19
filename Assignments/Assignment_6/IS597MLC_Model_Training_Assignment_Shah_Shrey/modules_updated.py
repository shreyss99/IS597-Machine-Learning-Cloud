#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[2]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[4]:


def load_data_new(filename, colname, log_file):
    """
    Read in input file and load data

    filename: csv file
    colname: column name for texts
    log_file: file object to write log messages
    return: X and y dataframe
    """

    ## 1. Read in data from input file
    df = pd.read_csv(filename, sep="\t", encoding='utf-8')
    
    log_file.write("************** Loading Data ************\n\n")

    # Check number of rows and columns
    log_file.write("No of Rows: {}\n".format(df.shape[0]))
    log_file.write("No of Columns: {}\n\n".format(df.shape[1]))

    ## 2. Select data needed for processing
    log_file.write(f"Selecting columns needed for processing: pmid, {colname}, rct\n\n")
    df = df[['pmid', colname, 'rct']]
    

    ## 3. Cleaning data
    # Trim unnecessary spaces for strings
    df[colname] = df[colname].apply(lambda x: str(x))

    # 3-1. Remove null values
    df=df.dropna()

    # Check number of rows and columns
    log_file.write("No of rows (After dropping null): {}\n".format(df.shape[0]))
    log_file.write("No of columns: {}\n\n".format(df.shape[1]))

    # 3-2. Remove duplicates and keep first occurrence
    df.drop_duplicates(subset=['pmid'], keep='first', inplace=True)

    # Check number of rows and columns
    log_file.write("No of rows (After removing duplicates): {}\n\n".format(df.shape[0]))

    # Check the first few instances
    log_file.write("<Data View: First Few Instances>\n\n")
    log_file.write(df.head(5).to_string(index=False) + "\n\n")
    
    # 3-3. Check label class
    log_file.write('Class Counts(label, row): Total\n')
    log_file.write(df["rct"].value_counts().to_string() + "\n\n")
    

    ## 4. Split into X and y (target)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    return X, y


# In[5]:


def split_data_new(X_data, y_data, log_file):
    """
    Read in the X_data and y_data and split into train, validation, and test sets.

    X_data: dataframe consisting of only the input features
    y_data: series consisting of only the output label
    log_file: file object to write log messages
    return: a tuple of the split data consisting of train, test and validation sets for both X_data and y_data
    """

    log_file.write("\n************** Splitting Data **************\n\n")

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    # Data Shape
    log_file.write("Train Data: {}\n".format(X_train.shape))
    log_file.write("Val Data: {}\n".format(X_val.shape))
    log_file.write("Test Data: {}\n\n".format(X_test.shape))

    # Label Distribution
    log_file.write('Class Counts(label, row): Train\n')
    log_file.write(y_train.value_counts().to_string() + "\n\n")
    log_file.write('Class Counts(label, row): Validation\n')
    log_file.write(y_val.value_counts().to_string() + "\n\n")
    log_file.write('Class Counts(label, row): Test\n')
    log_file.write(y_test.value_counts().to_string() + "\n\n")

    # Display the first 3 instances of X data
    log_file.write("Data View: X Train\n")
    log_file.write(X_train.head(3).to_string(index=False) + "\n\n")
    log_file.write("Data View: X Val\n")
    log_file.write(X_val.head(3).to_string(index=False) + "\n\n")
    log_file.write("Data View: X Test\n")
    log_file.write(X_test.head(3).to_string(index=False) + "\n\n")

    log_file.write("************** Resetting Index **************\n\n")

    # Reset index
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Data Shape after resetting index
    log_file.write("Train Data: {}\n".format(X_train.shape))
    log_file.write("Validation Data: {}\n".format(X_val.shape))
    log_file.write("Test Data: {}\n\n".format(X_test.shape))

    # Label Distribution after resetting index
    log_file.write('Class Counts(label, row): Train\n')
    log_file.write(y_train.value_counts().to_string() + "\n\n")
    log_file.write('Class Counts(label, row): Validation\n')
    log_file.write(y_val.value_counts().to_string() + "\n\n")
    log_file.write('Class Counts(label, row): Test\n')
    log_file.write(y_test.value_counts().to_string() + "\n\n")

    # Display the first 3 instances of X data after resetting index
    log_file.write("Data View: X Train\n")
    log_file.write(X_train.head(3).to_string(index=False) + "\n\n")
    log_file.write("Data View: X Val\n")
    log_file.write(X_val.head(3).to_string(index=False) + "\n\n")
    log_file.write("Data View: X Test\n")
    log_file.write(X_test.head(3).to_string(index=False) + "\n\n")

    return X_train, X_val, X_test, y_train, y_val, y_test


# In[6]:


def preprocess_data_new(X_data_raw, log_file):
    """
    Function to preprocess data with lowercase conversion, punctuation removal, tokenization, stemming

    X_data_raw: X data in dataframe
    log_file: file object to write log messages
    return: transformed dataframe
    """

    log_file.write("\n************** Pre-processed Data **************\n")

    X_data = X_data_raw.iloc[:, -1].astype(str)
    log_file.write(f"\nData Shape: {X_data.shape}\n")

    # 1. convert all characters to lowercase
    X_data = X_data.map(lambda x: x.lower())

    # 2. remove punctuation
    X_data = X_data.str.replace('[^\w\s]', '')

    # 3. tokenize sentence
    X_data = X_data.apply(nltk.word_tokenize)

    # 4. remove stopwords
    stopword_list = stopwords.words("english")
    X_data = X_data.apply(lambda x: [word for word in x if word not in stopword_list])

    # 5. stemming
    stemmer = PorterStemmer()
    X_data = X_data.apply(lambda x: [stemmer.stem(y) for y in x])

    # 6. removing unnecessary space
    X_data = X_data.apply(lambda x: " ".join(x))

    # Check data view
    log_file.write("\nData View:\n")
    log_file.write(X_data.head(3).to_string(index=False) + "\n")

    return X_data


# In[7]:


def fit_model_new(X, y, modelname, log_file):
    """
    Fits a machine learning model to input data.

    Parameters:
    X: Input features.
    y: Target variable.
    modelname: Name of the machine learning algorithm to use.
               Choose from: Decision Tree, Logistic Regression, Support Vector Machines, Random Forest.
    log_file: file object to write log messages
    """
    
    # Mapping modelname to corresponding machine learning algorithm
    models = {
        'Decision_tree': DecisionTreeClassifier(),
        'Logistic_regression': LogisticRegression(),
        'Support_vector_machine': SVC(),
        'Random_forest': RandomForestClassifier()
    }
    
    # Checking if the specified modelname is valid
    if modelname not in models:
        raise ValueError("Invalid modelname. Choose from: Decision_tree, Logistic_regression, Support_vector_machine, Random_forest.")
    
    # Fitting the selected model to the data
    
    log_file.write(f"\n************** Training Model: {modelname} **************\n")
    
    model = models[modelname]
    model.fit(X, y)
    
    return model


# In[8]:


def evaluate_model_new(y_pred, y_true, log_file):
    """
    Computes the confusion matrix for evaluating model performance.

    Parameters:
    y_pred Predicted labels.
    y_true: Actual labels.
    log_file: file object to write log messages

    Returns:
    array: Confusion matrix.
    """
    
    log_file.write("\n************** Model Evaluation **************\n")
    log_file.write("\nConfusion Matrix:\n")
    
    # Computing the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Writing the confusion matrix to the log file
    log_file.write(str(cm) + '\n')
    
    return cm


# In[ ]:




