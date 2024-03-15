from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB

import numpy as np
import pandas as pd
import time

import dataloader as dl

def ordinal_transform(df):

    # X 
    feature_vectors_df = dl.get_feature_vector_dataframe(df)
    
    # y
    labels_df = dl.get_labels_dataframe(df)

    # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features
    enc = OrdinalEncoder()

    # this gives an np array
    feature_vectors = feature_vectors_df.to_numpy()
    ordinal_feature_vectors = enc.fit_transform(feature_vectors_df)

    num_categories = np.array([len(x) for x in enc.categories_]).ravel()

    #print("categories_: ", enc.categories_)

    # need to go back to dataframe?
    ordinal_feature_vectors_df = pd.DataFrame(ordinal_feature_vectors)


    result = pd.concat([ordinal_feature_vectors_df, labels_df], axis=1)

    return result, num_categories





def main():

    dataframe, label_metrics, label_mapping = dl.load_df("/mnt/c/Users/aiant/phdspace/odobleja/odo-bot/odobot.db", "dataset-indigo-full", import_as_strings=True)

    ordinal_df, num_categories = ordinal_transform(dataframe)

    print(ordinal_df)

    train, test = dl.split(ordinal_df)

    X_train = dl.get_feature_vector_dataframe(train).to_numpy()
    y_train = dl.get_labels_dataframe(train).to_numpy().ravel()

    X_test = dl.get_feature_vector_dataframe(test).to_numpy()
    y_test = dl.get_labels_dataframe(test).to_numpy().ravel()

    print("shape of x: ", X_train.shape)
    print("shape of y: ", y_train.shape)

    model = CategoricalNB(min_categories=num_categories)
    st = time.time() # log start time
    model.fit(X_train, y_train)
    et = time.time() # log end time

    total_correct = 0
    total = 0

    # results = model.predict(test.iloc[:,:-1])
    # print(results)

    for index, row in test.iterrows():

        prediction = model.predict([row[:-1].to_numpy()])
        prediction = prediction[0]

        # Get the corresponding label metric
        metric = label_metrics[int(row["Decision"])]

        print("Predicted: ", prediction, " Actual: ", row["Decision"])

        if(row["Decision"] == prediction):
            metric.addCorrect()
            total_correct += 1
        else:
            metric.addIncorrect()
        
        total+=1
    
    print("Test results:")
    print("Total correct ", total_correct, " of ", total, " ", total_correct/total)

    elapsed_time = et - st
    print("Fit/train time: ", elapsed_time, ' seconds')

    dl.print_label_metrics(label_metrics)

    for index, log_prob in enumerate(model.feature_log_prob_):
        print(f"[{index}]\n", log_prob )

    print("model.score result: ")
    print(model.score(X_test, y_test))

if __name__ == '__main__':
    main()

