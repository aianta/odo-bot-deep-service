from flask import Flask
from flask.views import MethodView, View
import marshmallow as ma
from flask_smorest import Api, Blueprint, abort

from .model import Embedding, EmbeddingSet
from .schemas import EmbeddingRequestSchema, EmbeddingSchema, ActivityLabelsResponseSchema, ActivityLabelsRequestSchema
from .core import roberta_embeddings

import kmedoids
from sklearn.metrics.pairwise import euclidean_distances


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

blp_activity_label_generation = Blueprint(
    'activity-labels', 'activity-labels', url_prefix='/activitylabels',
    description="Generates unique activity labels for a set of TimelineEntities."
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
@blp_activity_label_generation.route('/')
class ActivityLabels(MethodView):

    @blp_activity_label_generation.arguments(ActivityLabelsRequestSchema)
    @blp_activity_label_generation.response(200,ActivityLabelsResponseSchema)
    def post(self, request):
        print("Processing activity labels request")
        embeddings_objects = [Embedding(id=x['id'], tensor=deep_model.embed(x['terms'], x['id'])) for x in request['entities']]
        _embeddings = [x.tensor for x in embeddings_objects]
        all_embeddings = np.vstack(_embeddings)

        distances = euclidean_distances(all_embeddings)

        result = kmedoids.fasterpam(diss=distances, medoids=50, max_iter=100, n_cpu=15)
        print(result.labels)


        results = [kmedoids.fasterpam(diss=distances, medoids=x, max_iter=100, n_cpu=15) for x in range(2,len(embeddings_objects)//2)]

        results.sort(key=lambda x: x.loss)

        results_summary = [x.loss for x in results]
        print(results_summary)

        print(results[0].labels, results[0].labels.shape)

        mapping_entries = [(request['entities'][index]['id'], int(cluster)) for index, cluster in enumerate(results[0].labels)]
        response = {
            "id": request['id'],
            "mappings":dict(mapping_entries),
            "cluster_info": {}
        }

        print('from ', len(request['entities']), ' produced ', np.unique(results[0].labels).shape[0], ' unique activity labels')

        return response 

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
api.register_blueprint(blp_activity_label_generation)

if __name__ == "__main__":
    app.run(host="0.0.0.0")