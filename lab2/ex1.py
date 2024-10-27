import pyod
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

a = 1.5 
b = 4 

def random_points(x_stdev, noise_stdev, N=1000):
    x_positions = np.random.normal(loc=0, scale=x_stdev, size=N)
    noise = np.random.normal(0, noise_stdev, size=N)
    y_positions = a * x_positions + b + noise

    X = np.stack((x_positions, y_positions, np.ones(N)), axis=-1) 
    return X

def compute_leverages(X):
   z = np.matmul(X.T, X)
   w = np.linalg.solve(z, X.T)
   H = np.matmul(X, w)
   leverages = [H[i, i] for i in range(len(X))]
   return leverages

def plot(points, title):
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    leverage_scores = compute_leverages(points)
    expected_value = np.float64(2/1000)

    distances = np.abs(leverage_scores - expected_value)
    threshold = np.quantile(distances, 0.8)

    is_anomaly = distances > threshold
    anomalous = points[is_anomaly]
    non_anomalous = points[~is_anomaly]

    plt.title(title)
    plt.axline((-10, -10 *a +b), (10, 10 * a + b))
    plt.scatter(non_anomalous[:, 0], non_anomalous[:, 1], color="g")
    plt.scatter(anomalous[:, 0], anomalous[:, 1], color="r")
    plt.show()

normal_points = random_points(x_stdev=8, noise_stdev=8) 
high_var_x= random_points(x_stdev=16, noise_stdev=8) 
high_var_y= random_points(x_stdev=8, noise_stdev=16) 
high_var_xy= random_points(x_stdev=16, noise_stdev=16) 

plot(normal_points, "Normal")
plot(high_var_x, "High var x")
plot(high_var_y, "High var y")
plot(high_var_xy, "High var xy")