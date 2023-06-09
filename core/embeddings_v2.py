
import torch
import numpy as np
from .utils import compute_centroid

class Embeddings_v2():

    def __init__(self, roberta):
        # Call super class constructor
        super().__init__()

        self.model = roberta.model


    '''
    Features should be a list of triples:
    (<feature_name>, <data>, <weight>)

    Where <data> is either a list of strings or a singular numeric value.

    Where <weight> is a decimal value co-efficient representing the weight of this
    feature in the final vector result.

    The 'event_type' parameter is used to define the type of underlying event being
    embedded. See the embedders dictionary in __init__ for valid options.
    '''
    def embed(self, features, count, total):
        print("embedding {}/{}".format(count, total))

        int_components = [self.processIntComponent(x) for x in features if type(x[1]) == int]
        float_components = [self.processFloatComponent(x) for x in features if type(x[1]) == float]
        string_components = [self.processListComponent(x) for x in features if type(x[1]) == list]

        print('int_components: {}'.format(int_components))
        print('float_components: {}'.format(float_components))
        print('string_components: {}'.format(string_components))

        '''
        https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
        '''
        fused_int_components = np.concatenate( int_components, axis=None) if len(int_components) > 0 else None
        fused_float_components = np.concatenate( float_components, axis=None) if len(float_components) > 0 else None
        fused_string_components = np.mean(np.vstack(string_components), axis=0) if len(string_components) > 0 else None

        print('fused_int_components: {}'.format(fused_int_components))
        print('fused_float_components: {}'.format(fused_float_components))
        print('fused_string_components: {}'.format(fused_string_components))

        component_list = []
        if fused_int_components is not None:
            component_list.append(fused_int_components)
        if fused_float_components is not None:
            component_list.append(fused_float_components)
        if fused_string_components is not None:
            component_list.append(fused_string_components)
    

        fused_embedding = np.concatenate(component_list, axis=None)
        print("{} fused embedding: {}".format(fused_embedding.shape, fused_embedding))

        # Normalize fused embedding to produce final embedding
        final_embedding = fused_embedding / np.linalg.norm(fused_embedding)

        print('{} final embedding: {}'.format(final_embedding.shape ,final_embedding))

        return final_embedding


    def processIntComponent(self, feature_component):
        print("computing feature embedding of {}. Type: int.".format(feature_component[0]))
        return np.array([feature_component[1]])

    def processFloatComponent(self, feature_component):
        print("computing feature embedding of {}. Type: float.".format(feature_component[0]))
        return np.array([feature_component[1]])

    def processListComponent(self, feature_component):

        print("computing feature embedding of {}. Type: List of strings.".format(feature_component[0]))
        terms = feature_component[1]

        sub_embeddings = self.embedListOfStrings(terms)
        embedding = compute_centroid(sub_embeddings)

        return embedding

    def embedListOfStrings(self, terms):

        # TODO: remove this once enhancements are complete or consider performance implications
        terms = terms[0:10]

        # Encode terms to roberta tokens 
        token_set = [self.model.encode(term) for term in terms]

        # Run the tokens through roberta
        layers = [self.model.extract_features(tokens, return_all_hiddens=True)[-1] for tokens in token_set]

        # Average the tensors for each token together to produce 1-d embedding for the whole term.
        # see: https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        embeddings = [torch.mean(layer, dim=1).squeeze() for layer in layers]

        # unbind the tensors from the gpu
        embeddings_cpu = [embedding.cpu().detach().numpy() for embedding in embeddings]

        return list(embeddings_cpu)

    

   