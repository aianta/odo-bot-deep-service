import marshmallow as ma

'''
Define Schemas
'''
class EmbeddingSchema(ma.Schema):
    id = ma.fields.UUID()
    # embedding = ma.fields.String() #TODO: figure out what type this should be.

class EmbeddingRequestSchema(ma.Schema):
    id = ma.fields.UUID() # Allow request to specify a uuid for convenience
    entity = ma.fields.Dict()