from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM 
from pyod.models.deep_svdd import DeepSVDD 
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt



def plot_outliers(fig, position, X, y, title):
    ax = fig.add_subplot(*position, projection='3d')
    ax.set_title(title)
    ax.scatter(X[:, 0][y], X[:, 1][y], X[:, 2][y], c="r")
    ax.scatter(X[:, 0][~y], X[:, 1][~y], X[:, 2][~y], c="g")



# Exercise 1
contamination = 0.15

X_train, X_test, y_train, y_test = generate_data(n_train=300, n_test=200, n_features=3, contamination=contamination)
y_train = y_train.astype(bool)
y_test =  y_test.astype(bool)

def benchmark_model(model):
    fig = plt.figure()
    model.fit(X_train)

    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    predictions_scores_train = model.decision_function(X_train)
    predictions_scores_test = model.decision_function(X_test)

    print(f"Balanced accuracy train =  {balanced_accuracy_score(y_train, predictions_train)}")
    print(f"Balanced accuracy test =  {balanced_accuracy_score(y_test, predictions_test)}")

    print(f"ROC auc score train =  {roc_auc_score(y_train, predictions_scores_train)}")
    print(f"ROC auc score test =  {roc_auc_score(y_test, predictions_scores_test)}")


    plot_outliers(fig, (2, 2, 1), X_train, y_train, "train true")
    plot_outliers(fig, (2, 2, 2), X_train, predictions_train.astype(bool), "train predicted")
    plot_outliers(fig, (2, 2, 3), X_test, y_test, "test true")
    plot_outliers(fig, (2, 2, 4), X_test, predictions_test.astype(bool), "test predicted")
    plt.show()

benchmark_model(OCSVM(contamination=contamination, kernel="linear"))
benchmark_model(OCSVM(contamination=contamination, kernel="rbf"))
benchmark_model(DeepSVDD(n_features=3, contamination=contamination))