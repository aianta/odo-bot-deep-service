from flask import Flask, send_file
from flask.views import MethodView, View
import marshmallow as ma
from flask_smorest import Api, Blueprint, abort
from flask import request

from .model import Embedding, EmbeddingSet
from .schemas import EmbeddingRequestSchema, EmbeddingSchema, ActivityLabelsResponseSchema, ActivityLabelsRequestSchema
from .core import roberta_embeddings

import kmedoids
import pm4py
from sklearn.metrics.pairwise import euclidean_distances
from kneed import KneeLocator


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

blp_make_model = Blueprint(
    'model', 'model', url_prefix='/model',
    description="Generates process model from xes file."
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

def organize_entities(entities):
    '''
    {
        <symbol>: [<entities with that symbol>],
        ...
    }
    '''
    entities_by_symbol = {}

    for entity in entities:
        symbol = entity['symbol']

        if symbol in entities_by_symbol:
            entity_list = entities_by_symbol[symbol]
            entity_list.append(entity)
        else:
            # Create the entity list for this symbol if it doesn't exist
            entity_list = []
            entities_by_symbol[symbol] = entity_list
            entity_list.append(entity)

    
    return entities_by_symbol

def compute_activity_mappings(entities, symbol):
    embeddings_objects = [Embedding(id=x['id'], tensor=deep_model.embed(x['terms'], x['id'])) for x in entities]
    _embeddings = [x.tensor for x in embeddings_objects]
    all_embeddings = np.vstack(_embeddings)
    
    print('all_embeddings shape:', all_embeddings.shape)

    distances = euclidean_distances(all_embeddings)

    '''
    If we send 3 objects for clustering 3//2 = 1 and therefore the range function or the clustering fails, so let's avoid that.
    '''
    if len(embeddings_objects) < 4:
        results = [kmedoids.fasterpam(diss=distances, medoids=x, max_iter=100, n_cpu=15) for x in range(2,len(embeddings_objects))]
    else:
        results = [kmedoids.fasterpam(diss=distances, medoids=x, max_iter=100, n_cpu=15) for x in range(2,len(embeddings_objects)//2)]

    '''
    Use knee method to determine optimal clustering
    https://pypi.org/project/kneed/
    https://towardsdatascience.com/detecting-knee-elbow-points-in-a-graph-d13fc517a63c

    Allegedly S=0 is best in an offline setting.
    '''
    losses = [x.loss for x in results]
    k_values = [i for i in range(2, len(results) + 2)]

    print("losses: ", losses)
    print("k_values:", k_values)

    kneedle = KneeLocator(k_values, losses,S=0, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow
    print("optimal K: ", optimal_k)

    #results.sort(key=lambda x: x.loss)
    optimal_result = results[optimal_k-2] # Optimal k index is optimal_k-2 because at index [0] k = 2
    
    print("optimal result: ", optimal_result.labels, optimal_result.labels.shape)


    results_summary = [x.loss for x in results]
    print(results_summary)


    mapping_entries = [(entities[index]['id'], symbol + "#" + str(int(cluster))) for index, cluster in enumerate(optimal_result.labels)]

    print('from ', len(entities), ' produced ', np.unique(optimal_result.labels).shape[0], ' unique activity labels')

    return dict(mapping_entries)


'''
API
'''
@blp_make_model.route('/')
class MakeModel(MethodView):

    def post(self):
        # Persist the xes file to a temporary location
        temp_file = open('temp.xes', "wb")
        print(request.data)
        temp_file.write(request.data)
        temp_file.flush()
        temp_file.close()

        eventlog = pm4py.read_xes("temp.xes")
        eventlog = pm4py.format_dataframe(eventlog, case_id="case:id", activity_key="activity", timestamp_key="timestamp")
        start_activities = pm4py.get_start_activities(eventlog)
        end_activities = pm4py.get_end_activities(eventlog)

        print("Start activities: {}\nEnd activities: {}".format(start_activities, end_activities))

        process_tree = pm4py.discover_process_tree_inductive(eventlog)
        bpmn_model = pm4py.convert_to_bpmn(process_tree)

        pm4py.save_vis_bpmn(bpmn_model, "out.png")
        pm4py.save_vis_process_tree(process_tree, "out_tree.png")

        return send_file(
            "out.png",
            "image/png"
        )




@blp_activity_label_generation.route('/')
class ActivityLabels(MethodView):

    @blp_activity_label_generation.arguments(ActivityLabelsRequestSchema)
    # @blp_activity_label_generation.response(200,ActivityLabelsResponseSchema)
    def post(self, request):
        print("Processing activity labels request")
        entities_by_symbol = organize_entities(request['entities'])

        print(entities_by_symbol.keys())

        response = {
            "id": request['id']
        }

        mappings = {}
        for symbol in entities_by_symbol.keys():
            print("Computing activity label mappings for symbol ", symbol)
            _mappings = compute_activity_mappings(entities_by_symbol[symbol], symbol)
            mappings.update(_mappings)

        response['mappings'] = mappings
        

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
api.register_blueprint(blp_make_model)

if __name__ == "__main__":
    app.run(host="0.0.0.0")