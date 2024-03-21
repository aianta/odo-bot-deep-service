import sqlite3
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from label_classification_metric import LabelClassificationMetric



def make_label_classification_metrics( db_path, dataset_name):

    metrics_result = {}
    label_mapping = {}


    query = f"SELECT extras FROM training_dataset WHERE dataset_name = '{dataset_name}';"

    # Create a connection to the database
    con = sqlite3.connect(db_path)

    cursor = con.cursor()

    for row in cursor.execute(query):
        extras = json.loads(row[0])
        
        metric = LabelClassificationMetric(extras["pathHash"], extras["path"])
        metrics_result[extras["pathHash"]] = metric
        label_mapping[extras["pathHash"]] = extras["path"]

    con.close()

    return metrics_result, label_mapping

def get_feature_vector_dataframe(df):
    return df.iloc[:,:-1]

def get_labels_dataframe(df):
    return df.iloc[:, -1:]

def split(df, test_size=0.1):

    # Remove data points belonging to a group of size fewer than 2 for a class. IE: if we only have 1 data point of a label, remove that row
    # Inspired from this answer on stackoverflow: https://stackoverflow.com/questions/53832858/drop-rows-corresponding-to-groups-smaller-than-specified-size
    df = df.groupby(df.columns[-1]).filter(lambda x: len(x) >= 2)


    feature_vectors_df = get_feature_vector_dataframe(df)
    labels_df = get_labels_dataframe(df)

    feature_vectors_train, feature_vectors_test, labels_train, labels_test = train_test_split(
        feature_vectors_df,
        labels_df,
        stratify=labels_df,
        test_size=test_size
    )


    train = pd.concat([feature_vectors_train, labels_train], axis=1)
    print("Train Dataset")
    print(train)

    test = pd.concat([feature_vectors_test, labels_test], axis=1)
    print("Test Dataset")
    print(test)

    return train, test

def print_label_metrics(label_metrics):
    print("---------------------------------------------")
    print("Path,hash,correct,incorrect,total,ratio,percent_correct,percent_incorrect")
    for key, value in label_metrics.items():
        print(value.human_label, ",",
            value.label, ",",
            value.correct, ",",
            value.incorrect, ",",
            value.total(), "," ,
            value.ratio(), ",",
            value.percent_correct(), ",",
            value.percent_incorrect(), ",",
            )

def load_df(db_path, dataset_name, import_as_strings=False):

    print(f"Loading database from file: {db_path}")

    metrics_result = {}
    label_mapping = {}

    # Create the connection to the database
    con = sqlite3.connect(db_path)

    cursor = con.cursor()

    dataset = []

    # The result of cursor.execute can be iterated over by row
    query = f"SELECT * FROM training_dataset WHERE dataset_name = '{dataset_name}';"
    print(query)
    data_point_size = -1
    for row in cursor.execute(query): # heptuples (size 7)
        feature_vector_string = row[2] # Feature vector stored as stringified json array is 3rd column (2nd index)
        labels_string = row[3] # Label vector stored as stringified json array is 4th column (3rd index)
        extras_string = row[6] # Extras stored as strngified JsonObject is 7th column (6th index) 


        feature_vector = json.loads(feature_vector_string)
        # Convert the feature vector values into the spcified datatype
        if import_as_strings:
            feature_vector = [str(x) for x in feature_vector] # Convert values to strings
        else:
            feature_vector = [np.float64(x) for x in feature_vector] #Convert strings to np.float64s

        labels = json.loads(labels_string) # 3 labels, has of API call, hash of request and hash of response all available as labels.
        extras = json.loads(extras_string)

        metric = LabelClassificationMetric(extras["pathHash"], extras["path"])
        metrics_result[extras["pathHash"]] = metric
        label_mapping[extras["pathHash"]] = extras["path"]

        '''
        Create a 'row' in our pandas dataframe. Should be:
        [label, 0, 1, 2, 3, ..., <feature_vector_length>] 
        '''
        data_point = [ *feature_vector,  labels[0] ]
        data_point_size = len(data_point)

        dataset.append(data_point)

    con.close()

    # Setup column names as just the index, except for the last column which will be named 'label'
    column_names = [str(x) for x in range(0,data_point_size-1)]
    column_names = [*column_names, 'Decision']
    df = pd.DataFrame(dataset, columns=column_names) 

    #df = df.iloc[:,-40:] #tiny subset for testing

    print(df)
    return df, metrics_result, label_mapping


