import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings

def train_eval(model, train_in, train_out, val_in, val_out):
    model.fit(train_in, train_out)
    pred_val = model.predict(val_in)
    #print("\nPredicted classes: ", pred_val, "\n")

    return accuracy_score(val_out, pred_val)


def main():
    data = pd.read_csv("dataset.csv")

    #print("Unique Class",data["emotion"].unique(),"\n")

    labels = data["emotion"]
    inputs = data.drop("emotion",axis=1)

    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    data_in, test_in, data_out, test_out = train_test_split(
        inputs, 
        labels, 
        test_size=0.1, 
        random_state=18, 
        stratify=labels)
    
    train_in, val_in, train_out, val_out = train_test_split(
        data_in,
        data_out,
        test_size=(0.2/0.9),
        random_state=18,
        stratify=data_out
    )
    #print("\nLenght of each split of the data: ", len(train_in), len(val_in), len(test_in), "\n")

    model_1 = SVC()
    print(
        "\nAccuracy of SVM: ",
        train_eval(model_1, train_in, train_out, val_in, val_out)
    )

    model_2 = KNeighborsClassifier()
    print(
        "\nAccuracy of KNN: ",
        train_eval(model_2, train_in, train_out, val_in, val_out)
    )

    model_3 = RandomForestClassifier()
    print(
        "\nAccuracy of RFC: ",
        train_eval(model_3, train_in, train_out, val_in, val_out)
    )

    # param_grid = [
    #     {"kernel": ["poly"], "degree": [3, 15, 25, 50]},
    #     {"kernel": ["rbf", "linear", "sigmoid"]}
    # ]

    # best_model = GridSearchCV(SVC(), param_grid)
    # best_model.fit(train_in, train_out)  # Fits on all combinations and keeps best model

    # print(
    #     "\n\nBest model with best parameters on test set: ",
    #     accuracy_score(
    #         test_out,
    #         best_model.predict(test_in)
    #     )
    # )
    # print(
    #     "Best parameters of best model: ",
    #     best_model.best_params_
    # )
    

if __name__ == "__main__":
    main()