from flask import Flask
from flask.views import MethodView, View
import marshmallow as ma
from flask_smorest import Api, Blueprint, abort

from .model import Embedding, EmbeddingSet
from .schemas import EmbeddingRequestSchema, EmbeddingSchema
from .roberta import roberta_embeddings

import uuid

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

# Define deep_model
deep_model = roberta_embeddings.RoBERTa(app)

embedding_sets = {}
embeddings = {}

'''
API
'''

@blp.route('/')
class Embeddings(MethodView):

    #@blp.response(200, ma.fields.Str())
    def get(self):
        return [str(x) for x in embeddings.keys()]

    @blp.arguments(EmbeddingRequestSchema, location='json')
    @blp.response(201,EmbeddingSchema)
    def post(self, request):
        print(request)

        # Extract entity from request body
        entity = request['entity']
        terms = entity['terms']

        # Create embedding object
        embedding = Embedding(tensor=deep_model.embed(terms))
        
        # Save embedding to embeddings dict
        embeddings[embedding.id] = embedding
        
        print(embeddings)

        return embedding


api.register_blueprint(blp)

