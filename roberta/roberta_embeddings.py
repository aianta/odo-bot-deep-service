import torch
import numpy as np
import sys

from .utils import compute_centroid



class RoBERTa:
    model = torch.hub.load('pytorch/fairseq', 'roberta.large')

    def __init__(self):
        '''
        For reproduceability. See: https://pytorch.org/docs/stable/notes/randomness.html
        '''
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 

        # Move to GPU
        print( "CUDA is available: " , torch.cuda.is_available())

        RoBERTa.model.to('cuda')
        RoBERTa.model.eval()

    '''
    Computes embeddings for all terms, returns a list of embeddings corresponding with each term. Probably (lol) preserving the order.
    TODO: map isn't pythonic, switch to list comprehensions sometime.
    '''
    def _compute_embeddings(self, terms):
        print("computing embeddings for ", len(terms), " terms.")

        # Encode terms to roberta tokens
        token_set = map (lambda term: RoBERTa.model.encode(term), terms)

        # Run the tokens through roberta
        layers = map (lambda tokens: RoBERTa.model.extract_features(tokens, return_all_hiddens=True)[-1], token_set)

        # Average the tensors for each token together to produce 1-d embedding for the whole term.
        # see: https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        embeddings = map(lambda layer: torch.mean(layer, dim=1).squeeze(), layers)

        # unbind the tensors from the gpu
        embeddings_cpu = map(lambda embedding: embedding.cpu().detach().numpy(), embeddings)

        return list(embeddings_cpu)

    '''
    Given:
    terms -  a list of strings representing an entity

    Returns:
    A 1-d embedding of the entity
    '''
    def embed(self, terms, id):

        embeddings = self._compute_embeddings(terms)
        embedding = compute_centroid(embeddings)
        
        #save our work
        with open('embeddings/' + id + ".npy", 'wb') as f:
            np.save(f, embedding)

        return embedding

