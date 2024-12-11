from pyod.models.pca import PCA
from pyod.models.kpca import KPCA 
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

# Loading the matlab file
data = loadmat("shuttle.mat")
X, y = data["X"], data["y"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

# Normalization
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Computing the contamination rate
contamination_rate = np.mean(y_train)

# Fitting the model.
model = PCA(contamination=contamination_rate)
model.fit(X_train)

# Plotting the individual variances.
variances = model.explained_variance_ratio_[::-1]
cumulative_variances = np.cumsum(variances)
plt.title("Individual variances")
plt.bar(range(1, len(variances)+1), variances)
plt.xticks(range(1, len(variances)+1))
plt.show()

plt.title("Cumulative variances")
plt.bar(range(1, len(variances)+1), cumulative_variances)
plt.xticks(range(1, len(variances)+1))

plt.show()
print(variances)
print(cumulative_variances)

# Making predictions.
predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

print(f"Balanced accuracy on the train set for PCA: {balanced_accuracy_score(y_train, predictions_train)}")
print(f"Balanced accuracy on the test set for PCA: {balanced_accuracy_score(y_test, predictions_test)}")


# Doing the same thing but for KPCA
# Notice that we decrease the train size because it takes too long.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = KPCA(contamination=contamination_rate)
contamination_rate = np.mean(y_train)
model.fit(X_train)

predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

print(f"Balanced accuracy on the train set for KPCA: {balanced_accuracy_score(y_train, predictions_train)}")
print(f"Balanced accuracy on the test set for KPCA: {balanced_accuracy_score(y_test, predictions_test)}")