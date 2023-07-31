from flask import Flask, send_file
from flask.views import MethodView, View
import marshmallow as ma
from flask_smorest import Api, Blueprint, abort
from flask import request

from .model import Embedding, EmbeddingSet
from .schemas import EmbeddingRequestSchema, EmbeddingSchema, ActivityLabelsResponseSchema, ActivityLabelsRequestSchema
from .core import roberta_embeddings
from .core.embeddings_v2 import Embeddings_v2

import kmedoids
import pm4py
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


import json
import uuid
import numpy as np
import os
import base64
import math

bpmn_output = "bpmn_out.png"
tree_output = "tree_out.png"
dfg_output = "dfg_out.png"
petri_net_output = "petri_net_inductive_out.png"
transition_system_output = "transition_system_out.png"

'''
NOTE: this is important! Odo bot's logpreprocessor looks for this prefix in the response
to properly persist the artifacts.
'''
fig_name_prefix = "clustering_results_"

# TODO: https://stackoverflow.com/questions/68619341/how-should-the-startup-of-a-flask-app-be-structured

app = Flask(__name__)
app.config['OPENAPI_VERSION'] = '3.0.2'
app.config['API_TITLE'] = 'Odo-bot Deep Service'
app.config['API_VERSION'] = 'v0.1'
api = Api(app)

blp_tfidf = Blueprint(
    'tfidf-activity-labels', 'tfidf-activity-labels', url_prefix='/activitylabels/v3',
    description='Generate unique activity labels for a set of events using the tfidf strategy.'
)

blp = Blueprint(
    'embeddings', 'embeddings', url_prefix='/embeddings',
    description='Embedding operations'
)

blp_distance = Blueprint(
    'distance', 'distance', url_prefix='/embeddings/distance',
    description='Computes the distances between computed embeddings.'
)

blp_activity_label_generation_v2 = Blueprint(
    'activity-labels-v2', 'activity-labels-v2', url_prefix='/activitylabels/v2',
    description="Generates unique activity labels for a set of TimelineEntities using enhanced embedding techniques."
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
embedding_logic_v2 = Embeddings_v2(deep_model) 


'''
Processes embedded objects into activity labels using clustering techniques.
'''
def _process_embeddings(entities, distances, symbol, data):
    
    
    '''
    K-medoids documentation:
    https://python-kmedoids.readthedocs.io/en/latest/

    If we send 3 objects for clustering 3//2 = 1 and therefore the range function or the clustering fails, so let's avoid that.
    '''
    if len(entities) < 4:
        results = [kmedoids.fasterpam(diss=distances, medoids=x, max_iter=100, n_cpu=15) for x in range(2,len(entities))]
    else:
        results = [kmedoids.fasterpam(diss=distances, medoids=x, max_iter=100, n_cpu=15) for x in range(2,len(entities)//2)]

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

    if len(losses) <= 2: # can't do knee on fewer than 3 data points
        # results.sort(key=lambda x: x.loss)
        # optimal_result = results[0]

        #TODO: this is hyper messy, please refactor
        mapping_entries = [(entities[index]['id'], symbol + "#0") for index in range(0,len(entities))]
        return dict(mapping_entries), None, None, None, None


    kneedle = KneeLocator(k_values, losses,S=0, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow 
    print("optimal K: ", optimal_k)

    print("length of losses array: ", len(losses))
    print("length of k_values array: ", len(k_values))
    
    # Choose optimal clustering and produce mapping result
    optimal_result = results[optimal_k-2] # Optimal k index is optimal_k-2 because at index [0] k = 2
    
    print("optimal result: ", optimal_result.labels, optimal_result.labels.shape)


    results_summary = [x.loss for x in results]
    print(results_summary)

    mapping_entries = [(entities[index]['id'], symbol + "#" + str(int(cluster))) for index, cluster in enumerate(optimal_result.labels)]

    print('from ', len(entities), ' produced ', np.unique(optimal_result.labels).shape[0], ' unique activity labels')

    return dict(mapping_entries), optimal_result, k_values, losses, optimal_k

def make_k_clustering_evaluation_figure(fig_name_prefix, symbol, k_values,losses, optimal_k):
    # Create a figure showing losses vs k-values. We'll use this for auditing/sanity checking. 
    fig_file_name = fig_name_prefix+symbol+".png"
    fig,ax = plt.subplots()
    #ax.plot(losses, k_values)
    ax.scatter(x=k_values, y=losses)
    ax.set(xlabel="# of clusters", ylabel="Loss (sum of deviations)")
    ax.set_yscale('log')
    ax.set_xticks(np.arange(min(k_values), max(k_values)+1, math.ceil(max(k_values)/10) if max(k_values) > 10 else 1.0 ))
    ax.axvline(x=optimal_k, color='b')
    plt.text(-0.1,-0.2, "Optimal K: " + str(optimal_k),transform=ax.transAxes, va='bottom', ha='left', wrap=True)
    
    fig.tight_layout() #avoid cutting off axis labels.
    
    fig.savefig(fig_file_name)
    fig.clear()

    return fig_file_name

def make_pca(figure_prefix, symbol, data, labels):
    '''
    Visualize PCA for analysis.
    https://machinelearningmastery.com/principal-component-analysis-for-visualization/

    Since we're normalizing vectors during the embedding process, I don't think we'll need to the standard scalar normalization here.
    '''
    pca_fig_name = figure_prefix + symbol + "_PCA.png"

    pca = PCA()
    pca_embeddings_t = pca.fit_transform(data)
    pca_fig = plt.figure("PCA figure")
    pca_plot = plt.scatter(pca_embeddings_t[:,0], pca_embeddings_t[:,1],c=labels)
    plt.title("PCA for ("+symbol+") Clustering")
    plt.legend(handles=pca_plot.legend_elements()[0], labels=list(np.unique(labels)))
    pca_ax = pca_plot.axes
    plt.text(-0.1,-0.2, "Top 5 VR: " + str(pca.explained_variance_ratio_[0:5]),transform=pca_ax.transAxes, va='bottom', ha='left', wrap=True)
    pca_fig.tight_layout()
    pca_fig.savefig(pca_fig_name)
    pca_fig.clear()
    
    print("VC: ", pca.explained_variance_ratio_)

    return pca_fig_name




def preprocess_entity(entity, symbol, weight_modifier=1.0):
    '''
    Need to get to [(<name>, <data>, weight)] form

    July 31 2023: Let's make dom effects positive or negative based on their action. 
    '''
    print(json.dumps(entity, indent=4))
    feature_list = []

    '''
    ClickEntry specific components
    '''
    if symbol == 'CE':
        if 'localizedTerms' in entity and len(entity['localizedTerms']) > 0:
            feature_list.append(('localizedTerms', entity['localizedTerms'], 1.0 * weight_modifier))
        else:
            # only consider terms if 'localized terms' are not available.
            if 'terms' in entity and len(entity['terms']) > 0:
                feature_list.append(('terms', entity['terms'], 1.0 * weight_modifier))

        feature_list = addListComponentIfExists('cssClassTerms', 1.0 * weight_modifier, feature_list, entity)
        feature_list = addListComponentIfExists('idTerms', 1.0 * weight_modifier, feature_list, entity)

    '''
    Effect specific components
    '''
    if symbol == 'E':
        
        # feature_list = addListComponentIfExists('madeVisible', 1.0, feature_list, entity)
        # feature_list = addListComponentIfExists('madeInvisible', -1.0, feature_list, entity)

        feature_list = addListComponentIfExists('terms_added', 1.0 * weight_modifier, feature_list, entity)
        feature_list = addListComponentIfExists('terms_removed', -1.0 * weight_modifier, feature_list, entity) # Note the negative weight on removed terms.
        feature_list = addListComponentIfExists('cssClassTerms_added', 1.0 * weight_modifier, feature_list, entity)
        feature_list = addListComponentIfExists('cssClassTerms_removed', -1.0 * weight_modifier, feature_list, entity) # Note the negative weight on removed terms.
        feature_list = addListComponentIfExists('idTerms_added', 1.0 * weight_modifier, feature_list, entity)
        feature_list = addListComponentIfExists('idTerms_removed', -1.0 * weight_modifier, feature_list, entity) # Note the negative weight on removed terms.

    '''
    Data Entry specific components
    '''
    if symbol == 'DE':

        if 'localizedTerms' in entity and len(entity['localizedTerms']) > 0:
            feature_list.append(('localizedTerms', entity['localizedTerms'], 1.0 * weight_modifier))
        else:
            # only consider terms if 'localized terms' are not available.
            if 'terms' in entity and len(entity['terms']) > 0:
                feature_list.append(('terms', entity['terms'], 1.0 * weight_modifier))  


        feature_list = addListComponentIfExists('cssClassTerms', 1.0 * weight_modifier, feature_list, entity)
        feature_list = addListComponentIfExists('idTerms', 1.0 * weight_modifier, feature_list, entity)      


    '''
    General components 
    '''

    # if 'size' in entity:
    #     feature_list.append(('size', entity['size'], 1.0))
    


    if 'previous' in entity and entity['previous']:
        previous_features = preprocess_entity(entity['previous'], entity['previous']['symbol'], 0.0)
        feature_list = feature_list + previous_features

    if 'next' in entity and entity['next']:
        next_features = preprocess_entity(entity['next'], entity['next']['symbol'], 0.5)    
        feature_list = feature_list + next_features

    return feature_list

def addListComponentIfExists(component_name, weight, feature_list, entity):
    if component_name in entity and len(entity[component_name]) > 0:
        feature_list.append((component_name, entity[component_name], weight))
    return feature_list

def dist(v1, v2):
    return float(np.linalg.norm(v1 - v2))

'''
Takes in a list of events and returns a list of documents. 
Documents have an id corresponding to the source event id and a data field
populated by concatenating all terms for an event with spaces.
'''
def events_to_documents(event_list):


    documents = [{
        "id": event['id'],
        "data": event_to_document(event)
        } for event in event_list]

    return documents

def event_to_document(event):
    return " ".join(event['terms']) if 'terms' in event else "" + " ".join(event['cssClassTerms']) if 'cssClassTerms' in event else "" + " ".join(event['idTerms']) if 'idTerms' in event else "" + " ".join(event['localizedTerms']) if 'localizedTerms' in event else "" + event_to_document(event['previous']) if 'previous' in event and event['previous'] else "" + event_to_document(event['next']) if 'next' in event and event['next'] else "" 



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
        dfg, start, end = pm4py.discover_dfg(eventlog)
        petri_net, initial, final = pm4py.discover_petri_net_inductive(eventlog)
        transition_system = pm4py.discover_transition_system(eventlog)

        pm4py.save_vis_bpmn(bpmn_model, bpmn_output)
        pm4py.save_vis_process_tree(process_tree, tree_output)
        pm4py.save_vis_dfg(dfg, start, end, dfg_output)
        pm4py.save_vis_petri_net(petri_net, initial, final, petri_net_output)
        pm4py.save_vis_transition_system(transition_system, transition_system_output)

        with open(bpmn_output, 'rb') as bpmn_file, open(tree_output, 'rb') as tree_file, open(dfg_output, 'rb') as dfg_file, open(petri_net_output, 'rb') as petri_file, open(transition_system_output, 'rb') as transition_file:
            bpmn_bytes = bpmn_file.read()
            tree_bytes = tree_file.read()
            dfg_bytes = dfg_file.read()
            petri_bytes = petri_file.read()
            transition_bytes = transition_file.read()

            response = {
                "bpmn": base64.b64encode(bpmn_bytes).decode('utf-8'),
                "tree": base64.b64encode(tree_bytes).decode('utf-8'),
                "dfg":  base64.b64encode(dfg_bytes).decode('utf-8'),
                "petri": base64.b64encode(petri_bytes).decode('utf-8'),
                "transition": base64.b64encode(transition_bytes).decode('utf-8')
            }

            return response




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

            embeddings_objects = [Embedding(id=x['id'], tensor=deep_model.embed(x['terms'], x['id'])) for x in entities_by_symbol[symbol]]
    
            _embeddings = [x.tensor for x in embeddings_objects]
            all_embeddings = np.vstack(_embeddings)

            print('all_embeddings shape:', all_embeddings.shape)

            _distances = euclidean_distances(all_embeddings)

            _mappings, optimal_result, k_values, losses, optimal_k = _process_embeddings(entities_by_symbol[symbol], _distances, symbol, all_embeddings)
           
            fig_file_name = make_k_clustering_evaluation_figure(fig_name_prefix, symbol, k_values, losses, optimal_k)
            pca_file_name = make_pca(fig_name_prefix, symbol, all_embeddings, optimal_result.labels )

            mappings.update(_mappings)

            if fig_file_name is not None:
            # add clustering elbow chart to reponse
                with open(fig_file_name, 'rb') as fig_file: 
                    fig_file_bytes = fig_file.read()

                    response['clustering_results_' + symbol] = base64.b64encode(fig_file_bytes).decode('utf-8')
            # add PCA chart to response
            if pca_file_name is not None:
                with open(pca_file_name, 'rb') as pca_file:
                    pca_file_bytes = pca_file.read()

                    response['clustering_results_' + symbol + '_PCA'] = base64.b64encode(pca_file_bytes).decode('utf-8')

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

@blp_tfidf.route('/')
class TFIDFActivityLabels(MethodView):

    @blp_tfidf.arguments(ActivityLabelsRequestSchema)
    def post(self, request):
        print("Processing activity labeling request with TF-IDF strategy.")
        events_by_symbol = organize_entities(request['entities'])

        response = {
            "id":request["id"]
        }

        mappings = {}
        for symbol in events_by_symbol.keys():
            
            # Init TF-IDF
            tfidf = TfidfVectorizer(
                min_df=4,
                max_df=0.95,
                max_features=8000,
                stop_words='english'
            )

            documents = events_to_documents(events_by_symbol[symbol])
            documents = [document['data'] for document in documents]
            tfidf.fit(documents)
            text = tfidf.transform(documents)

            print('TF-IDF transform() ouput for ', symbol, ": ", text)

            print("TF-IDF output shape: ", text.shape)

            print("TF-IDF typeof: ", type(text))

            print('Number of events/documents: ', len(events_by_symbol[symbol]))

            distances = euclidean_distances(text)

            print("Distances: ", distances)

            print("distances shape: ", distances.shape)

            _mappings, optimal_result, k_values, losses, optimal_k = _process_embeddings(events_by_symbol[symbol], distances, symbol, text)

            fig_file_name = make_k_clustering_evaluation_figure(fig_name_prefix, symbol, k_values, losses, optimal_k)

            mappings.update(_mappings)

            if fig_file_name is not None:
            # add clustering elbow chart to reponse
                with open(fig_file_name, 'rb') as fig_file: 
                    fig_file_bytes = fig_file.read()

                    response['clustering_results_' + symbol] = base64.b64encode(fig_file_bytes).decode('utf-8')
        

        response['mappings'] = mappings

        return response




@blp_activity_label_generation_v2.route('/')
class EnhancedEmbeddings(MethodView):

    @blp_activity_label_generation_v2.arguments(ActivityLabelsRequestSchema)
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

            # Convert entities to triple form
            preprocessed_entitites = [ (entity['id'],preprocess_entity(entity, symbol)) for entity in entities_by_symbol[symbol] ]
            
            embeddings_objects = [Embedding(id=entry[0], tensor=embedding_logic_v2.embed(features=entry[1], count=count,total=len(preprocessed_entitites) )) for count,entry in enumerate(preprocessed_entitites)]
            
            _embeddings = [x.tensor for x in embeddings_objects]
            all_embeddings = np.vstack(_embeddings)
            print('all_embeddings shape:', all_embeddings.shape)
            _distances = euclidean_distances(all_embeddings)

            _mappings, optimal_result, k_values, losses, optimal_k = _process_embeddings(entities_by_symbol[symbol],_distances, symbol, all_embeddings)
            
            mappings.update(_mappings)

            # Additional diagnostic data only exists if there were enough entities (2) to do knee method operations.
            if optimal_result is not None and k_values is not None and losses is not None and optimal_k is not None:

                fig_file_name = make_k_clustering_evaluation_figure(fig_name_prefix, symbol, k_values, losses, optimal_k)
                pca_file_name = make_pca(fig_name_prefix, symbol, all_embeddings, optimal_result.labels)
                
                

                if fig_file_name is not None:
                # add clustering elbow chart to reponse
                    with open(fig_file_name, 'rb') as fig_file: 
                        fig_file_bytes = fig_file.read()

                        response['clustering_results_' + symbol] = base64.b64encode(fig_file_bytes).decode('utf-8')

                # add PCA chart to response
                if pca_file_name is not None:
                    with open(pca_file_name, 'rb') as pca_file:
                        pca_file_bytes = pca_file.read()

                        response['clustering_results_' + symbol + '_PCA'] = base64.b64encode(pca_file_bytes).decode('utf-8')

        response['mappings'] = mappings
        

        return response 








# api.register_blueprint(blp)
api.register_blueprint(blp_distance)
api.register_blueprint(blp_activity_label_generation)
api.register_blueprint(blp_activity_label_generation_v2)
api.register_blueprint(blp_tfidf)
api.register_blueprint(blp_make_model)


if __name__ == "__main__":
    app.run(host="0.0.0.0")