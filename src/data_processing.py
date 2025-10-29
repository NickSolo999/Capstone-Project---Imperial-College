"""
Simple helper functions for PVS data analysis and machine learning.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from IPython.display import display, HTML
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def getDatasets(base_path='../data'):
    """
    Returns all loaded datasets from PVS 1-9.

    Args:
        base_path (str): Base path to the data directory. Defaults to '../data' for notebooks.

    Returns:
        dict: Dictionary containing data_left, data_right, and data_labels for each PVS dataset
    """
    pvs = {}

    for i in range(1, 10):
        folder = os.path.join(base_path, "PVS " + str(i))
        data_left = pd.read_csv(os.path.join(folder, 'dataset_gps_mpu_left.csv'), float_precision="high")
        data_right = pd.read_csv(os.path.join(folder, 'dataset_gps_mpu_right.csv'), float_precision="high")
        data_labels = pd.read_csv(os.path.join(folder, 'dataset_labels.csv'))

        pvs["pvs_" + str(i)] = {
            "data_left": data_left,
            "data_right": data_right,
            "data_labels": data_labels
        }

    return pvs


def plotDataClass(pvs, classes, datasets):
    """
    Shows data classes on a plot.

    Args:
        pvs (int): PVS dataset number (1-9)
        classes (list): List of class names to plot
        datasets (dict): Dictionary of datasets from getDatasets()
    """
    data_labels = datasets["pvs_" + str(pvs)]["data_labels"]
    plt.figure(figsize=(16, 6))

    for i in range(0, len(classes)):
        classe = classes[i]
        (data_labels[classe] * (i+1)).plot(linewidth=2)

    plt.legend()


def createMapDataClass(pvs, classes, colors, datasets, zoom_start=14):
    """
    Shows data classes on a map.

    Args:
        pvs (int): PVS dataset number (1-9)
        classes (list): List of class names to display
        colors (list): List of colors for each class
        datasets (dict): Dictionary of datasets from getDatasets()
        zoom_start (int): Initial zoom level for the map

    Returns:
        str: HTML string of the rendered map
    """
    dataset = datasets["pvs_" + str(pvs)]
    data = pd.concat([dataset["data_left"], dataset["data_labels"]], axis=1)

    gps = data[['latitude', 'longitude']]
    focolat = (gps['latitude'].min() + gps['latitude'].max()) / 2
    focolon = (gps['longitude'].min() + gps['longitude'].max()) / 2
    maps = folium.Map(location=[focolat, focolon], zoom_start=zoom_start)

    grouper = data.groupby(["latitude", "longitude"]).mean().round(0)

    for i in range(0, len(classes)):
        classe = classes[i]
        color = colors[i]
        points = grouper[grouper[classe] == 1].index.values.reshape(-1)

        for point in points:
            folium.Circle(point, color=color, radius=0.1).add_to(maps)

    return maps.get_root().render().replace('"', '&quot;')


def createLegendDataClass(classes_names, colors):
    """
    Shows legend for data classes.

    Args:
        classes_names (list): List of class names for the legend
        colors (list): List of colors corresponding to each class

    Returns:
        str: HTML string of the legend
    """
    html_legend = """
    <style>
        .legend { list-style: none; }
        .legend li { float: left; margin-right: 10px; }
        .legend span { border: 1px solid #ccc; float: left; width: 12px; height: 12px; margin: 2px; }
    </style>
    <div style="width: 100%; height: 10px;">
    <ul class="legend" style="list-style: none;">
    """

    for i in range(0, len(classes_names)):
        name = classes_names[i]
        color = colors[i]

        html_legend += """
        <li><span style="background-color: {}"></span> {}</li>
        """.format(color, name)

    html_legend += """
    </ul>
    </div>
    <br>
    """

    return html_legend


def showMapDataClass(pvs, classes, classes_names, colors, datasets):
    """
    Shows data class maps side by side.

    Args:
        pvs (int or list): PVS dataset number(s) (1-9)
        classes (list): List of class names to display
        classes_names (list): List of display names for classes
        colors (list): List of colors for each class
        datasets (dict): Dictionary of datasets from getDatasets()
    """
    html = createLegendDataClass(classes_names, colors)

    html += """
    <div>
    """

    if isinstance(pvs, list):
        for i in pvs:
            maps = createMapDataClass(i, classes, colors, datasets, 13)
            html += """
            <iframe srcdoc="{}" style="float:left; width: {}px; height: {}px; display:inline-block; width:33%; margin: 0 auto; border: 1px solid black"></iframe>
            """.format(maps, 500, 500)
    else:
        maps = createMapDataClass(pvs, classes, colors, datasets)
        html += """
            <iframe srcdoc="{}" style="float:left; width: 99%; height: 500px; display:inline-block; margin: 0 auto; border: 1px solid black"></iframe>
            """.format(maps)

    html += "</div>"

    display(HTML(html))


def metricsDataClass(classes, datasets):
    """
    Measure the quantity and distribution metrics of the data classes.

    Args:
        classes (list): List of class names to analyze
        datasets (dict): Dictionary of datasets from getDatasets()

    Returns:
        pd.DataFrame: DataFrame with class counts and distribution percentages
    """
    list_data = []

    for pvs in range(1, 10):
        data = datasets["pvs_" + str(pvs)]
        list_data.append(data["data_labels"][classes].sum())

    data = pd.DataFrame(list_data)
    data["Total"] = data.sum(axis=1)

    for classe in classes:
        data[classe + "_distribuition_%"] = round(data[classe]/data["Total"] * 100, 2)

    data.index = np.arange(1, len(data) + 1)
    data.index = data.index.rename("PVS")
    return data


def create_statistical_features(data, sensor_columns, window_size=50):
    """
    Create statistical features (mean, std, variance) for sensor data.

    Args:
        data (pd.DataFrame): Input sensor data
        sensor_columns (list): List of sensor column names
        window_size (int): Window size for rolling statistics (default: 50)

    Returns:
        pd.DataFrame: DataFrame with statistical features only
    """
    features = pd.DataFrame(index=data.index)

    for col in sensor_columns:
        if col in data.columns:
            features[f'{col}_mean'] = data[col].rolling(window=window_size, min_periods=1).mean() # create rolling window mean. min_periods=1 - calculate mean even if there are less than window_size values (avoids NaN at the beginning)
            features[f'{col}_std'] = data[col].rolling(window=window_size, min_periods=1).std()
            features[f'{col}_var'] = data[col].rolling(window=window_size, min_periods=1).var()

    return features


def perform_kmeans_clustering(data, n_clusters=3):
    """
    Perform K-means clustering on the data.

    Args:
        data (pd.DataFrame): Data for clustering
        n_clusters (int): Number of clusters (default: 3)

    Returns:
        tuple: (cluster_labels, kmeans_model)
    """
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.fillna(0)) 
    # Fill NaN values with 0 for scaling
    # .fit() - compute the mean and std to be used by StandardScaler (z-score normalization)
    # .transform() - apply scaling using the computed mean and std, z = (x - mean) / std
    # zero mean, standard deviation of 1

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    # 1. .fit(scaled_data) - Training phase:
    # Initializes K random centroids (cluster centers)
    # Iteratively:
    # Assigns each data point to nearest centroid
    # Recalculates centroids as the mean of assigned points
    # Repeats until convergence (centroids stop moving significantly)

    # 2. .predict(scaled_data) - Prediction phase:
    # Assigns each data point to its closest cluster
    # Returns cluster labels (0, 1, 2, ... K-1) - array of integers indicating which cluster each point belongs to

    return cluster_labels, kmeans
