import numpy as np
import matplotlib.pyplot as plt

def plot_3d(points, title):
    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_title(title)

def plot_3d_with_labels(points, title, labels):
    ax = plt.figure().add_subplot(projection="3d")

    anomalous = points[labels]
    non_anomalous = points[~labels]
    ax.scatter(non_anomalous[:, 0], non_anomalous[:, 1], non_anomalous[:, 2], c="green")
    ax.scatter(anomalous[:, 0], anomalous[:, 1], anomalous[:, 2], c="red")
    ax.set_title(title)

def center_points(points):
    mean_axes = np.mean(points, axis=0)

    return points - mean_axes

def covariance_matrix(points):
    N = len(points)
    return (points.T @ points) / N

def evd(points):
    cov = covariance_matrix(points)
    return np.linalg.eigh(cov)

def anomalies_by_deviation_on_component(transformed_points, component):
    mean = np.mean(transformed_points[:, component])

    deviation = np.abs(transformed_points[:, component] - mean)
    deviation_threshold = np.quantile(deviation, 0.9) 

    anomalies = deviation > deviation_threshold
    return anomalies

def anomalies_by_distance(transformed_points, p, delta):
    centroid = np.mean(transformed_points, axis=0)
    d = len(p)

    distances = []

    for point in transformed_points:
        total = 0
        vec = point - centroid

        for i in range(d):
            dist = np.linalg.norm(np.dot(vec, p[i])) ** 2
            total += dist / delta[i]

        distances.append(total)
    
    distances = np.array(distances)
    distance_threshold = np.quantile(distances, 0.9)

    anomalies = distances > distance_threshold 
    return anomalies


mean = [5, 10, 2]
cov = [[3, 2, 2], [2, 10, 1], [2, 1, 2]]

points = np.random.multivariate_normal(mean, cov, size=500)
centered_points = center_points(points)
plot_3d(centered_points, "Centered points")
plt.show()

delta, p = evd(centered_points)
explained_variance = delta / np.sum(delta)

plt.title("Explained variance")
plt.bar(x=[1, 2, 3], height=explained_variance)
plt.xticks([1,2,3])
plt.show()

plt.title("Cumulative explained variance")
plt.bar(x=[1,2,3], height=np.cumsum(explained_variance))
plt.xticks([1,2,3])
plt.show()

transformed_points = centered_points @ p
transformed_points /= np.sqrt(delta)
plot_3d(transformed_points, "Projected")
plt.show()

anomalies = anomalies_by_deviation_on_component(transformed_points, 0)
plot_3d_with_labels(centered_points, "Anomalies for the third component", anomalies)
anomalies = anomalies_by_deviation_on_component(transformed_points, 1)
plot_3d_with_labels(centered_points, "Anomalies for the second component", anomalies)
anomalies = anomalies_by_deviation_on_component(transformed_points, 2)
plot_3d_with_labels(centered_points, "Anomalies for the first component", anomalies)
plt.show()

anomalies = anomalies_by_distance(transformed_points, p, delta)
plot_3d_with_labels(centered_points, "Anomalies by distance", anomalies)
plt.show()