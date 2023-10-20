from .odo_distance_metric import odo_distance_function
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def do_clustering( symbol, all_embeddings, embeddings_objects, _labels ):

    condensed_distance_matrix = pdist(all_embeddings, metric='euclidean')
    
    # Combine embedding distances with domain rules to create better clusterings
    updated_distance_matrix, max_distance = odo_distance_function(condensed_distance_matrix, embeddings_objects)

    linkage_data = linkage(updated_distance_matrix, method='single', metric='euclidean' )
    
    '''
    Using max distance as the cutoff for the number of clusters will always give us 1 cluster in the case where no_merge rules have been applied.
    thus we need to use ever so slightly less than the max distance to get the correct number of clusters.
    so let's do that by subtracting 1% from the max distance.
    '''
    max_distance = max_distance - (max_distance * 0.01)

    #cluster_file_name = make_dendrogram(fig_name_prefix, symbol, linkage_data, _labels, max_distance)
    print( "number of embedded objects: ", len(embeddings_objects), " symbol: ", symbol)
    # Pull out clusterings with k clusters where k varies from 2, to the number of embeddings objects.
    clusterings = [(fcluster(linkage_data, k, criterion='maxclust'), k) for k in range(2, len(embeddings_objects))]

    print("number of clusterings: ", len(clusterings))

    # Only keep clusterings that partition the data into at least 2 clusters. 
    clusterings = [cluster for cluster in clusterings if len(np.unique(cluster[0])) > 1]

    print("number of clusterings with at least 2 clusters: ", len(clusterings))

    # Compute the silhouette score for each clustering
    clusterings = [(silhouette_score(squareform(updated_distance_matrix), labels=cluster[0], metric='precomputed'), cluster[0], cluster[1]) for cluster in clusterings]
    # Sort the clusterings by silhouette score.
    clusterings.sort(key=lambda x: x[0], reverse=True) # Sort by silhouette score in descending order
    
    # Print the silhouette scores of the clusterings
    for cluster in clusterings:
        print("k=", cluster[2], " silhouette_score=", cluster[0])

    # If no clusterings could be found (ie: all candidates could not merge with each other)
    if len(clusterings) == 0:
        clusters = np.arange(0, len(embeddings_objects)) # put all candidates in their own cluster
    else:
        clusters = clusterings[0][1] # Select the top clustering

    # pca_file_name = make_pca(fig_name_prefix, symbol, all_embeddings, clusters )

    # mapping_entries = [(embeddings_objects[index].id, symbol+"#"+str(cluster)) for index, cluster in enumerate(clusters)]

    print("from ", len(embeddings_objects), "entities, we have ", len(np.unique(clusters)) , "unique activity labels.")

    return clusters, linkage_data, max_distance

def make_dendrogram(figure_prefix, symbol, data, labels, max_distance):
    dendrogram_fig_name = figure_prefix + symbol + "_hierarchical_clustering_dendrogram.png"

    figure = plt.figure(0, figsize=(20,8))
    fancy_dendrogram(data, labels=labels, orientation='right',max_d=max_distance)
    figure.tight_layout()
    figure.savefig(dendrogram_fig_name)
    figure.clear()
    return dendrogram_fig_name

'''
Shamelessly copied from: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/#Eye-Candy
Then modified for horizontal dendrograms
'''
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.ylabel('sample index')
        plt.xlabel('distance')
        for i, d, c in zip( ddata['dcoord'], ddata['icoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = 0.5 * sum(d[1:3])
            if x > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % x, (x,y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axvline(x=max_d, c='k')
    return ddata