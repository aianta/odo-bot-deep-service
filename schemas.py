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