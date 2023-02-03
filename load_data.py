import pandas as pd
from google.cloud import firestore
import streamlit as st
import openai
from openai.embeddings_utils import get_embedding
import numpy as np

from tqdm import tqdm
import logging

from sklearn.neighbors import NearestNeighbors
from functools import lru_cache


#IMPORTANT VARIABLES TO BE USED
service_account_info = st.secrets["gcp_service_account_firestore"]
openai.api_key = st.secrets['openai']['key']

def custom_key_function(input):
    if type(input) == pd.Series:
        input = tuple(input.iteritems())
    return input

@lru_cache(maxsize=None)
def load_firestore_client(service_account_info = service_account_info):
  firestore_client = firestore.Client.from_service_account_info(service_account_info)
  return firestore_client

firestore_client = load_firestore_client() #Carrega a conexão com a base de dados com cache.

@st.experimental_memo
def firestore_query(_firestore_client = firestore_client, field_paths = [], collection = 'tesauro'):
  #Load dataframe for code search
  firestore_collection = _firestore_client.collection(collection)
  filtered_collection = firestore_collection.select(field_paths)#Fields containing useful data for search engine
  filtered_collection = filtered_collection.get() #Returns a list of document snapshots, from which data can be retrieved
  filtered_collection_dict = [doc.to_dict() for doc in filtered_collection] #Returns list of dictionaries 
  filtered_collection_dataframe = pd.DataFrame.from_records(filtered_collection_dict) #Returns dataframe
  return filtered_collection_dataframe


@lru_cache(maxsize=None)
def load_logger():
    # instantiate logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # define handler and formatter
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    # add formatter to handler
    handler.setFormatter(formatter)

    # add handler to logger
    logger.addHandler(handler)

    return logger

logger = load_logger()

@lru_cache(maxsize=None)
def iterate(collection_name, batch_size=1000, batch_number=0, cursor=None):
    query = firestore_client.collection(collection_name).limit(batch_size).order_by('__name__')
    logger.info(f"Currently on batch number {batch_number}")
    if cursor:
        query = query.start_after(cursor)

    for doc in query.stream():
        yield doc
    
    if 'doc' in locals():
        batch_number += 1
        yield from iterate(collection_name, batch_size, batch_number=batch_number, cursor=doc)

@lru_cache(maxsize=None)
def load_icpc_embeddings_from_firestore():    
    logger.info("Preparing generator...")
    docs = iterate("expressions_embeddings_icpc")
    
    logger.info(f"Retrieving and converting embeddings query to a list of python dictionaries...")
    docs_dict = [doc.to_dict() for doc in tqdm(docs)] #Returns list of dictionaries 

    logger.info("Converting list of dictionaries containing embeddings to a DataFrame...")
    docs_df = pd.DataFrame.from_records(docs_dict) #Returns dataframe
    
    logger.info("ICPC embeddings DataFrame is ready!")
    return docs_df


@lru_cache(maxsize=None)
def load_icpc_embeddings_from_hdf():
    return pd.read_hdf('data\\tesauro_embeddings.h5', 'embeddings')

icpc_embeddings_df = load_icpc_embeddings_from_hdf()

@lru_cache(maxsize=None)
def load_KNN_model(n_neighbors=5):
    # Initialize the NearestNeighbors class with the number of neighbors to search for
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)

    # Fit the data to the NearestNeighbors class
    embeddings = np.vstack(icpc_embeddings_df.to_list())
    nbrs.fit(embeddings)

    return nbrs


@lru_cache(maxsize=None)
def get_input_embedding(input):
    input_vector = get_embedding(input, engine="text-embedding-ada-002")
    return input_vector

tesauro_embeddings = load_icpc_embeddings_from_hdf()
tesauro_expressions = pd.read_hdf('data\\tesauro_embeddings.h5', 'expressions')
tesauro_codes = pd.read_hdf('data\\tesauro_embeddings.h5', 'code')
nbrs = load_KNN_model()

# Functions that gets the query and database with ICPC codes and related expressions and pre-loaded KNN model and returns a results dataframne
@lru_cache(maxsize=None)
def get_tesauro_query_results(input: str) -> pd.DataFrame:
    search_term_vector = get_input_embedding(input)

    # Use the kneighbors method to get the nearest neighbors
    distances, indices = nbrs.kneighbors([search_term_vector])

    # Organizing results for visualization and grouping by code
    results = []
    
    for index in indices[0]:
        icpc_code = tesauro_codes.iloc[index]
        expression = tesauro_expressions.iloc[index]

        row = {
            'code': icpc_code,
            'expression': expression
        }
        results.append(row)
    results_df = pd.DataFrame.from_dict(results)
    return results_df