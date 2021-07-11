import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

# define a AdaBoost classifier
clf = [AdaBoostClassifier(),GaussianNB()]
j=0

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    lengths = len(clf)
    acc =[]
    for i in range(lengths):
        # load the dataset from the official sklearn datasets
        X, y = datasets.load_iris(return_X_y=True)

        # do the test-train split and train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf[i].fit(X_train, y_train)

        # calculate the print the accuracy score
        accuracy = accuracy_score(y_test, clf[i].predict(X_test))
        acc.append(accuracy)
        print(f"Model trained with accuracy: {round(accuracy, 3)}")
    j = np.argmax(acc)
    print(j)
    print(acc)
    


# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf[j].predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf[j].fit(X, y)
