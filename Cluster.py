import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import numpy as np
import scipy.optimize as opt


def read_data(file_paths, selected_country, start_year, end_year):
    dataframes_list = []

    for path in file_paths:
        file_name = path.split('/')[-1].split('.')[0]
        df = pd.read_csv(path, skiprows=4)
        df = df.rename(columns={'Country Name': 'Country'})
        df = df.set_index('Country')
        df_selected_country = df.loc[selected_country, str(start_year):str(end_year)].transpose().reset_index()
        df_selected_country = df_selected_country.rename(columns={'index': 'Year', selected_country: file_name})
        dataframes_list.append(df_selected_country)

    # Concatenate all DataFrames based on the 'Year' column
    result_df = pd.concat(dataframes_list, axis=1)

    # Replace null values with the mean of each column
    result_df = result_df.apply(lambda col: col.fillna(col.mean()))

    return result_df


def scale_data(df):
    # Apply Standard Scaling to all columns in the DataFrame
    scaler = StandardScaler()
    result_df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return result_df_scaled


def plot_scatter_matrix(df):
    # Scatter plot
    pd.plotting.scatter_matrix(df, figsize=(9.0, 9.0))
    plt.tight_layout()
    plt.show()


def plot_heatmap(df):
    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Create a heatmap using Seaborn
    plt.figure(figsize=(15, 15))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation different indicator of Debt')
    # Rotate x-axis labels to 45 degrees
    plt.xticks(rotation=0, fontweight='bold')
    plt.yticks(rotation=0, fontweight='bold')

    # Save the plot with 300 dpi
    plt.savefig('Correlation.png', dpi=300)
    plt.show()


def cluster_and_plot(df, columns, n_clusters, save_filename):
    df_cluster = df[columns]

    # Calculate silhouette score for 2 to n_clusters
    for ic in range(2, n_clusters + 1):
        score = one_silhouette(df_cluster, ic)
        print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(df_cluster)
    # Extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    plt.figure(figsize=(6.0, 6.0))
    # Scatter plot with colours selected using the cluster numbers
    plt.scatter(df_cluster[columns[0]], df_cluster[columns[1]], c=labels, cmap="tab10")
    # Colour map Accent selected to increase contrast between colours
    # Show cluster centres
    xc = cen[:, 0]
    yc = cen[:, 1]
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    # c = colour, s = size
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.title(f"Cluster between {columns[0]} & {columns[1]}")
    # Save the plot with 300 dpi
    plt.savefig(save_filename, dpi=300)
    plt.show()


def one_silhouette(xy, n):
    """ Calculates silhouette score for n clusters """
    # Set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)  # Fit done on x,y pairs
    labels = kmeans.labels_
    # Calculate the silhouette score
    score = skmet.silhouette_score(xy, labels)
    return score


# Example usage
file_paths = ['IMF credit.csv',
              "Total reserves.csv", 'Personal remittances.csv',
              'Foreign investment.csv', 'ICT  exports.csv']
selected_country = "Pakistan"
start_year = 1960
end_year = 2022

result_df = read_data(file_paths, selected_country, start_year, end_year)

# Remove the 'Year' column
result_df = result_df.drop('Year', axis=1)

# Scale the data
result_df_scaled = scale_data(result_df)

print(result_df_scaled)

# Plot scatter matrix
plot_scatter_matrix(result_df_scaled)

# Plot correlation heatmap
plot_heatmap(result_df)

# Cluster and plot Personal remittances vs ICT exports
cluster_and_plot(result_df_scaled, ['Personal remittances', 'ICT  exports'], 2, 'Clusterremitence.png')

# Cluster and plot IMF credit vs ICT exports
cluster_and_plot(result_df_scaled, ['IMF credit', 'ICT  exports'], 2, 'ClusterIMF.png')
