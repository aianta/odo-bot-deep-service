
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

'''
Given a list of embeddings, compute their centroid. 
'''
def compute_centroid(embeddings):

    # stack the embedded vectors (tensors?) into an array
    all_embeddings = np.vstack(embeddings)
    centroid = np.mean(all_embeddings, axis=0)

    return centroid

def make_pca(figure_prefix, symbol, data, labels):
    '''
    Visualize PCA for analysis.
    https://machinelearningmastery.com/principal-component-analysis-for-visualization/

    Since we're normalizing vectors during the embedding process, I don't think we'll need to the standard scalar normalization here.
    '''
    pca_fig_name = figure_prefix + symbol + "_PCA.png"

    pca = PCA()
    pca_embeddings_t = pca.fit_transform(data)
    pca_fig = plt.figure("PCA figure")
    pca_plot = plt.scatter(pca_embeddings_t[:,0], pca_embeddings_t[:,1],c=labels)
    plt.title("PCA for ("+symbol+") Clustering")
    plt.legend(handles=pca_plot.legend_elements()[0], labels=list(np.unique(labels)))
    pca_ax = pca_plot.axes
    plt.text(-0.1,-0.2, "Top 5 VR: " + str(pca.explained_variance_ratio_[0:5]),transform=pca_ax.transAxes, va='bottom', ha='left', wrap=True)
    pca_fig.tight_layout()
    pca_fig.savefig(pca_fig_name)
    pca_fig.clear()
    
    print("VC: ", pca.explained_variance_ratio_)

    return pca_fig_name