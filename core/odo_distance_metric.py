import numpy as np
from scipy.spatial.distance import squareform

def odo_distance_function(condensed_distance_matrix, embeddings):

    dist_square = squareform(condensed_distance_matrix)
    max_dist = np.max(dist_square)
    no_merge_distance= max_dist * 2

    # with statement for pretty printing
    with np.printoptions(precision=3, suppress=True):
        print("dist_square:\n{}\n".format(dist_square))
        print("max_dist: {}".format(max_dist))
        print("no_merge_distance: {}".format(no_merge_distance))

    # For each pair of embeddings, compute the distance between them.
    # TODO - this is a very inefficient way to do this.
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            descision = decision_function(embeddings[i], embeddings[j])

            if descision == 'must_merge':
                dist_square[i][j] = 0
                dist_square[j][i] = 0
            elif descision == 'must_not_merge':
                dist_square[i][j] = no_merge_distance
                dist_square[j][i] = no_merge_distance
            elif descision == 'can_merge':
                # dist_square[i][j] = dist_square[i][j]
                # dist_square[j][i] = dist_square[j][i] 
                continue
    
    # Convert square distance matrix back to condensed distance matrix.
    return squareform(dist_square), no_merge_distance


def decision_function(entity1, entity2):
    
    # If the two entities have different baseURIs then they must not merge.
    if entity1.metadata['baseURI'] != entity2.metadata['baseURI']:
        return 'must_not_merge'

    # If the two entities are DEs and one follows the other then they must not merge. This is because summarization ensures these would be different DEs.
    if entity1.metadata['symbol'] == 'DE' and entity2.metadata['symbol'] == 'DE' and entity1.metadata['nextId'] == entity2.metadata['id']:
        return 'must_not_merge'
    
    merge = 'can_merge'

    merge = if_it_exists_it_must_match('idTerms', entity1, entity2)
    if merge != 'can_merge': # if we can still merge keep going
        return merge
    
    merge = if_it_exists_it_must_match('cssClassTerms', entity1, entity2)
    if merge != 'can_merge':
        return merge
        
    
    return merge


def if_it_exists_it_must_match(field, entity1, entity2):
    # If one entity has id terms, the other must have the same id term to merge.
    if field in entity1.metadata and field not in entity2.metadata:
        return 'must_not_merge' # do not merge if entity 1 has id terms but entity2 does not.
    if field in entity2.metadata and field not in entity1.metadata:
        return 'must_not_merge' # do not merge if entity 2 has id terms but entity1 does not.
    if field in entity1.metadata and field in entity2.metadata:
        if len(entity1.metadata[field]) != len(entity2.metadata[field]):
            return 'must_not_merge' # do not merge if entity 1 and entity 2 have different number of id terms. Incidently this should never be bigger than 1.
        if len(entity1.metadata[field]) == 1: # if there are id terms
            if len(entity1.metadata[field]) == 1 and entity1.metadata[field][0] == entity2.metadata['idTerms'][0]:
                return 'must_merge' # if the entities have the same id term then they must merge
            else:
                return 'must_not_merge' #if they have different id terms they must not merge

    return 'can_merge'        