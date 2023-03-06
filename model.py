import uuid

'''
Represents a single embedding
'''
class Embedding:
    
    def __init__(self, tensor):
        self.id = uuid.uuid4()
        self.tensor = tensor

'''
Aggregates collections of embeddings, to perform group operations
such as computing the distances between all embeddings in the set.
'''
class EmbeddingSet:

    def __init__(self, embeddings=None):
        # Init embedings list
        self.embeddings = set() if embeddings is None else embeddings 
