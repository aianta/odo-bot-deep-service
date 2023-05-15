# Deep Service

This is a lightweight flask application meant to provide deep learning functionality for odo bot. 

Currently the service provides following functionality:

* Compute embeddings for terms. 
* Compute centroids for term sets.
* Compute distances between centroids. 
* Compute unique activity labels for entity timelines

## Usage

Activate the conda environment (`roberta` in my case) and execute: 

`flask run --host=0.0.0.0`

Also requires graphviz for process model visualization, install with:

`sudo apt-get install graphviz`