import numpy as np
from .utils import compute_centroid

class ClickEventEmbeddings:

    def __init__(self, controller) -> None:
        # Need access to the RoBERTa model
        self.model = controller.model
        self.controller = controller
        
    '''
    Feature components should be triples: 
    (<feature_name>, <data>, <weight>)

    Where <data> is either a list of strings or a singular numeric value.

    Where <weight> is a decimal value co-efficient representing the weight of this
    feature in the final vector result.
    '''
