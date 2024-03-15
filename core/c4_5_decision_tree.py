import dataloader as dl
from chefboost import Chefboost as chef
import time
import argparse



def main():
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_out")
    args = parser.parse_args()

    model_output_file = args.model_out + ".pkl"

    print("model_out: ", model_output_file )



    label_metrics, label_mapping = dl.make_label_classification_metrics("/mnt/c/Users/aiant/phdspace/odobleja/odo-bot/odobot.db", "dataset-indigo-full")

    for key,value in label_mapping.items():
        print(value, " -> ", key)

    dataframe = dl.load_df("/mnt/c/Users/aiant/phdspace/odobleja/odo-bot/odobot.db", "dataset-indigo-full")

    train, test = dl.split(dataframe)

    config = {'algorithm': 'C4.5',
            'enableParallelism': False,
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