import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from pyod.models.iforest import IForest
from pyod.models.dif import DIF 
from pyod.models.loda import LODA 

# Generating the data.
X_train, y_train = make_blobs(n_samples=[500,500], n_features=3, centers=[(0, 10, 0), (10, 0, 10)], cluster_std=[1, 1])
X_test = np.random.uniform([-10, -10, -10], [20, 20, 20], size=(1000, 3)) 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title("The training set")
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2])
plt.show()

# Fitting models.
contamination_rate = 0.02
models = [(IForest(contamination=contamination_rate), "IForest"),
          (DIF(contamination=contamination_rate), "DIF"), 
          (DIF(contamination=contamination_rate, hidden_neurons=[128, 64, 32]), "DIF, neurons=[128, 64, 32]"), 
          (LODA(contamination=contamination_rate, n_bins=40, n_random_cuts=100), "LODA, n_bins=40, n_cuts=100"),
          (LODA(contamination=contamination_rate, n_bins=200, n_random_cuts=500), "LODA, n_bins=200, n_cuts=500"),
]

for (i, (model, name)) in enumerate(models):
    model.fit(X_train)
    scores = model.decision_function(X_test)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(name)
    ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=scores)
    plt.show()