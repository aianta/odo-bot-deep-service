from flask import Flask
from flask.views import MethodView, View
import marshmallow as ma
from flask_smorest import Api, Blueprint, abort

from .model import Embedding, EmbeddingSet
from .schemas import EmbeddingRequestSchema, EmbeddingSchema
from .roberta import roberta_embeddings

import uuid
import numpy as np
import os

# TODO: https://stackoverflow.com/questions/68619341/how-should-the-startup-of-a-flask-app-be-structured

app = Flask(__name__)
app.config['OPENAPI_VERSION'] = '3.0.2'
app.config['API_TITLE'] = 'Odo-bot Deep Service'
app.config['API_VERSION'] = 'v0.1'
api = Api(app)

blp = Blueprint(
    'embeddings', 'embeddings', url_prefix='/embeddings',
    description='Embedding operations'
)

blp_distance = Blueprint(
    'distance', 'distance', url_prefix='/embeddings/distance',
    description='Computes the distances between computed embeddings.'
)

# Define deep_model
deep_model = roberta_embeddings.RoBERTa()

# Declare embeddings dict
embeddings = {}

# Load existing embeddings from folder
embeddings_base_path = 'embeddings/'
for entry in os.listdir(embeddings_base_path):
    id = entry.split('.')[0]

    with open('embeddings/' + id + '.npy', 'rb') as f:
        tensor = np.load(f) 
        embeddings[id] = Embedding(id=id, tensor=tensor)

def dist(v1, v2):
    return float(np.linalg.norm(v1 - v2))
'''
API
'''
@blp_distance.route('/')
class Distance(MethodView):
    def get(self):

        all_distances = [(embedding1.id, embedding2.id, dist(embedding1.tensor, embedding2.tensor)) for embedding1 in embeddings.values() for embedding2 in embeddings.values() ]
        print(all_distances)

        response = {}

        # Convert the above into something JSONy that's convenient for odo-ui
        for triple in all_distances:
            if triple[0] not in response:
                response[triple[0]]={}

            response[triple[0]][triple[1]] = triple[2]

        return response

@blp.route('/')
class Embeddings(MethodView):

    #@blp.response(200, ma.fields.Str())
    def get(self):
        return [str(x) for x in embeddings.keys()]

    @blp.arguments(EmbeddingRequestSchema, location='json')
    @blp.response(201,EmbeddingSchema)
    def post(self, request):
        print(request)

        if(request['id'] not in embeddings):
            # Extract termsfrom request body
            terms = request['terms']

            # Create embedding object
            embedding = Embedding(id=request['id'],tensor=deep_model.embed(terms, request['id']))
            
            # Save embedding to embeddings dict
            embeddings[embedding.id] = embedding
            

            return embedding
        else:
            embedding = embeddings[request['id']]
            return embedding




api.register_blueprint(blp)
api.register_blueprint(blp_distance)

