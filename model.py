import uuid

'''
Represents a single embedding
'''
class Embedding:
    
    def __init__(self, id=None, tensor=None):
        self.id = uuid.uuid4() if id is None else id
        self.tensor = tensor
        print("embedding: " , tensor)

'''
Aggregates collections of embeddings, to perform group operations
such as computing the distances between all embeddings in the set.
'''
class EmbeddingSet:

    def __init__(self, embeddings=None):
        # Init embedings list
        self.embeddings = set() if embeddings is None else embeddings 
