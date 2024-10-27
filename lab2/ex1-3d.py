import pyod
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

a = 1.5 
b =  4 
c = 2

def compute_leverages(X):
    z = np.matmul(X.T, X)
    w = np.linalg.solve(z, X.T)
    H = np.matmul(X, w)
    leverages = [H[i, i] for i in range(len(X))]
    return leverages

def random_points(x1_stdev, x2_stdev, noise_stdev, N=1000):
    x1_positions = np.random.normal(loc=0, scale=x1_stdev, size=N)
    x2_positions = np.random.normal(loc=0, scale=x2_stdev, size=N)
    noise = np.random.normal(0, noise_stdev, size=N)

    y_positions = a * x1_positions + b*x2_positions + c + noise

    X = np.stack((x1_positions, x2_positions, y_positions, np.ones(N)), axis=-1) 
    return X

def plot(points):
    leverage_scores = compute_leverages(points)
    expected_value = np.float64(2/1000)

    distances = np.abs(leverage_scores - expected_value)
    threshold = np.quantile(distances, 0.8)

    is_anomaly = distances > threshold
    anomalous = points[is_anomaly]
    non_anomalous = points[~is_anomaly]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)


    ax.scatter(non_anomalous[:, 0], non_anomalous[:, 1], non_anomalous[:, 2], color="g")
    ax.scatter(anomalous[:, 0], anomalous[:, 1], anomalous[:, 2], color="r")
    plt.show()


normal_points = random_points(x1_stdev=8, x2_stdev=8, noise_stdev=16, N=1000) 
plot(normal_points)
