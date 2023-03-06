from flask import Flask
from flask.views import MethodView, View
import marshmallow as ma
from flask_smorest import Api, Blueprint, abort

from .model import Embedding, EmbeddingSet
from .schemas import EmbeddingRequestSchema, EmbeddingSchema


app = Flask(__name__)
app.config['OPENAPI_VERSION'] = '3.0.2'
app.config['API_TITLE'] = 'Odo-bot Deep Service'
app.config['API_VERSION'] = 'v0.1'
api = Api(app)

blp = Blueprint(
    'embeddings', 'embeddings', url_prefix='/embeddings',
    description='Embedding operations'
)



'''
API
'''

@blp.route('/')
class Embeddings(MethodView):

    @blp.arguments(EmbeddingRequestSchema, location='json')
    @blp.response(201,EmbeddingSchema)
    def post(self, request):
        print(request)
        embedding = Embedding(None)
        return embedding


api.register_blueprint(blp)