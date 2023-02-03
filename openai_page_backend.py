import pandas as pd
import streamlit as st
import openai
import numpy as np
import logging

from tqdm import tqdm
from google.cloud import firestore
from sklearn.neighbors import NearestNeighbors
from openai.embeddings_utils import get_embedding
from load_data import *

placeholder = st.empty() # Placeholder for important messages



#IMPORTANT VARIABLES TO BE USED
service_account_info = st.secrets["gcp_service_account_firestore"]
openai.api_key = st.secrets['openai']['key']


@st.cache(hash_funcs={firestore.Client: id}, ttl=None, show_spinner=True)
def load_firestore_client(service_account_info = service_account_info):
  firestore_client = firestore.Client.from_service_account_info(service_account_info)
  return firestore_client

firestore_client = load_firestore_client() #Carrega a conexão com a base de dados com cache.

@st.cache(hash_funcs={firestore.Client: id}, ttl=None, show_spinner=True, allow_output_mutation=True)
def firestore_query(firestore_client = firestore_client, field_paths = [], collection = 'tesauro'):
  #Load dataframe for code search
  firestore_collection = firestore_client.collection(collection)
  filtered_collection = firestore_collection.select(field_paths)#Fields containing useful data for search engine
  filtered_collection = filtered_collection.get() #Returns a list of document snapshots, from which data can be retrieved
  filtered_collection_dict = [doc.to_dict() for doc in filtered_collection] #Returns list of dictionaries 
  filtered_collection_dataframe = pd.DataFrame.from_records(filtered_collection_dict) #Returns dataframe
  return filtered_collection_dataframe

@st.experimental_memo
def get_code_criteria(code: str) -> dict[str, str]:
  code_criteria = ciap_criteria[ciap_criteria['code']==code].iloc[0].to_dict()
  return code_criteria

ciap_criteria = firestore_query(field_paths=['code','`inclusion criteria`', '`exclusion criteria`'], collection='ciap_criteria')
tesauro = firestore_query(field_paths=['`CIAP2_Código1`', '`Termo Português`'])
ciap_df = firestore_query(field_paths=['`CIAP2_Código1`', '`titulo original`']).drop_duplicates()


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

@st.experimental_memo(show_spinner=False, suppress_st_warning=True, experimental_allow_widgets=True)
def load_icpc_embeddings():    
    logger.info("Preparing generator...")
    docs = iterate("expressions_embeddings_icpc")
    
    logger.info(f"Retrieving and converting embeddings query to a list of python dictionaries...")
    docs_dict = [doc.to_dict() for doc in tqdm(docs)] #Returns list of dictionaries 

    logger.info("Converting list of dictionaries containing embeddings to a DataFrame...")
    docs_df = pd.DataFrame.from_records(docs_dict) #Returns dataframe
    
    logger.info("ICPC embeddings DataFrame is ready!")
    return docs_df

placeholder.info(f"We're preparing everything for you. Be patient. This can take minutes, but it's just for the first page load :)", icon='🤖')
icpc_embeddings_df = load_icpc_embeddings()
placeholder.success("Hurray! Everything is ready!", icon="😀")

@st.experimental_singleton
def load_KNN_model():
    # Initialize the NearestNeighbors class with the number of neighbors to search for
    nbrs = NearestNeighbors(n_neighbors=5)

    # Fit the data to the NearestNeighbors class
    embeddings = np.vstack(icpc_embeddings_df['similarities'].to_list())
    nbrs.fit(embeddings)

    return nbrs

nbrs = load_KNN_model()

@st.experimental_memo
def get_input_embedding(input):
    input_vector = get_embedding(input, engine="text-embedding-ada-002")
    return input_vector

st.header('CIAP2 e OpenAI')
st.write('Digite abaixo a condição clínica que deseja codificar e nós encontraremos para você os melhores códigos CIAP2.')        
st.text_input('Digite aqui o motivo de consulta', key="icpc_search_input", label_visibility='collapsed')
if st.session_state['icpc_search_input'] != "":
    search_term_vector = get_input_embedding(st.session_state['icpc_search_input'])

    # Use the kneighbors method to get the nearest neighbors
    distances, indices = nbrs.kneighbors([search_term_vector])

    # Organizing results for visualization and grouping by code
    results_df = pd.DataFrame(columns=['code', 'title', 'expression'])
    for index in indices[0]:
        icpc_code = icpc_embeddings_df.iloc[index]['CIAP2_Código1']
        icpc_code_title = ciap_df[ciap_df["CIAP2_Código1"]==icpc_code]['titulo original'].iloc[0]
        expression = icpc_embeddings_df.iloc[index]['Termo Português']
        
        row = pd.DataFrame.from_dict([{
            'code': icpc_code,
            'title': icpc_code_title,
            'expression': expression
        }])

        results_df = pd.concat([results_df, row])

    results_df = results_df.groupby(['code'], as_index = False, sort=False).agg({'title': 'first', 'expression': ' | '.join})
    st.write(f"Resultados encontrados para: **{st.session_state['icpc_search_input']}**")
    for row in results_df.to_dict('records'):
      with st.expander(f"__{row['code']} - {row['title']}__"):
            st.write(f"_{row['expression']}_")
            code_criteria = get_code_criteria(row['code'][0:3])
            st.write(f"**criterios de inclusão:** {code_criteria['inclusion criteria']}")
            st.write(f"**criterios de exclusão:** {code_criteria['exclusion criteria']}")
