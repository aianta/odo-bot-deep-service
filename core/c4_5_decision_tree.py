import dataloader as dl
from chefboost import Chefboost as chef
import time
import argparse
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_out")
    args = parser.parse_args()

    model_output_file = args.model_out + ".pkl"

    print("model_out: ", model_output_file )

    dataframe, label_metrics, label_mapping = dl.load_df("/mnt/c/Users/aiant/phdspace/odobleja/odo-bot/odobot.db", "dataset-indigo-full")

    features = dl.get_feature_vector_dataframe(dataframe)
    labels_df = dl.get_labels_dataframe(dataframe)

    features = MinMaxScaler().fit_transform(features)

    # Plotting explained variance ratio
    # https://www.kdnuggets.com/2023/05/principal-component-analysis-pca-scikitlearn.html
    # var_ratio = []
    # nums = list(range(1, 320, 30))
    # total_time = 0
    # pca_full = None
    # for num in nums:
    #     pca_st = time.time()
    #     print("Computing PCA for: ", num)
    #     pca_full = PCA(n_components=num, svd_solver="full")
    #     pca_full.fit(features)
    #     var_ratio.append(np.sum(pca_full.explained_variance_ratio_))
    #     pca_et = time.time()
    #     elapsed_time = pca_et - pca_st
    #     print("Time to compute PCA: ", elapsed_time, " seconds")
    #     total_time += elapsed_time
    
    # print("total PCA analysis time: ", total_time, " seconds")
    

    # variance_ratio_fig = plt.figure('ExplainVarianceRatio')
    # plt.grid()
    # plt.plot(nums, var_ratio, marker='o')
    # plt.xlabel('n_components')
    # plt.ylabel('Explained variance ratio')
    # plt.title('n_components vs Explained Variance Ratio')
    # plt.xticks(nums)
    # variance_ratio_fig.savefig('./Explained_variance_ratio_vs_n_components.png')
    # variance_ratio_fig.clear()

    pca_full = PCA(n_components=300, svd_solver="full")
    pca_features = pca_full.fit_transform(features)

    print("pca_features.shape: ",pca_features.shape)
    
    dataframe = pd.concat([pd.DataFrame(pca_features), labels_df], axis=1)

    print("Post Dimensionality reduction dataset:")
    print(dataframe)
   

    for key,value in label_mapping.items():
        print(value, " -> ", key)

     


    train, test = dl.split(dataframe)

    config = {'algorithm': 'C4.5',
            'enableParallelism': False,
            'enableRandomForest': True,
            'num_of_trees': 128,
            'num_cores': 4
            }
    st = time.time()
    model = chef.fit(train, config, target_label='Decision', validation_df=test)
    et = time.time()



    print("Saving model")
    chef.save_model(model, model_output_file)

    print("Computing label classification metrics on test dataset...")
    total_correct = 0
    total = 0

    print("Unique test labels",test['Decision'].unique())

    for index, row in test.iterrows():
    
        prediction = chef.predict(model, row[:-1])

        # Get the corresponding label metric
        metric = label_metrics[int(row["Decision"])]

        print("Predicted: ", prediction, " Actual: ", row["Decision"])

        if(row["Decision"] == prediction):
            metric.addCorrect()
            total_correct+=1
        else:
            metric.addIncorrect()
        
        total+=1

    print("Test results: ")
    print("Total correct ", total_correct, " of ", total, " ", total_correct/total)

    elapsed_time = et - st
    print("Fit/Train time: ", elapsed_time, ' seconds')

    dl.print_label_metrics(label_metrics)
        
# Sky falls in multi processing otherwise.
if __name__  == "__main__":
    main()