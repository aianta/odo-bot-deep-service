import marshmallow as ma
from marshmallow import INCLUDE

'''
Define Schemas
'''
class EmbeddingSchema(ma.Schema):
    id = ma.fields.String() # We're expecting something like <timeline_uuid>#<index>
    # embedding = ma.fields.String() #TODO: figure out what type this should be.

class EmbeddingRequestSchema(ma.Schema):
    class Meta:
        unknown = INCLUDE
    id = ma.fields.String() # Allow request to specify a uuid for convenience, this will be used to create an embedding set
    terms = ma.fields.List(ma.fields.String()) 

class ActivityLabelsRequestSchema(ma.Schema):
    class Meta:
        unknown = INCLUDE
    id = ma.fields.String() #Allow requests to sepcify a uuid for this activity labeling request.
    entities = ma.fields.List(ma.fields.Dict())

class ActivityLabelsResponseSchema(ma.Schema):
    class Meta:
        unknown = INCLUDE
    id = ma.fields.String() 
    # mappings = ma.fields.Dict() #An object that returns key-value pairs where the keys are timeline entity ids and the values are cluster ids.
    # cluster_info = ma.fields.Dict() #An object with key value pairs where the keys are cluster ids and the values are information about that cluster.