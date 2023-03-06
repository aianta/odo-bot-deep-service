

import numpy as np

'''
Given a list of embeddings, compute their centroid. 
'''
def compute_centroid(embeddings):

    # stack the embedded vectors (tensors?) into an array
    all_embeddings = np.vstack(embeddings)
    centroid = np.mean(all_embeddings, axis=0)

    return centroid
