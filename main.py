# To-do:
# 1. Do the BM25 ranking algorithm after openai embeddings search as this may rank better most relevant entries

import psutil
import os

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

import time 

placeholder = st.empty() # Placeholder for important messages



#IMPORTANT VARIABLES TO BE USED
service_account_info = st.secrets["gcp_service_account_firestore"]
openai.api_key = st.secrets['openai']['key']


@st.experimental_memo
def get_code_criteria(code: str) -> dict[str, str]:
  code_criteria = ciap_criteria[ciap_criteria['code']==code].iloc[0].to_dict()
  return code_criteria

@st.experimental_memo
def get_code_title(code: str) -> str:
  code_title = ciap_df[ciap_df['CIAP2_Código1']==code].iloc[0].to_dict()['titulo original']
  return code_title

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



#placeholder.info(f"We're preparing everything for you. Be patient. This can take minutes, but it's just for the first page load :)", icon='🤖')
icpc_embeddings_df = load_icpc_embeddings_from_hdf()
#placeholder.success("Hurray! Everything is ready!", icon="😀")



st.header('CIAP2 e OpenAI')
st.write('Digite abaixo a condição clínica que deseja codificar e nós encontraremos para você os melhores códigos CIAP2.')        
st.text_input('Digite aqui o motivo de consulta', key="icpc_search_input", label_visibility='collapsed')
if st.session_state['icpc_search_input'] != "":
    search_start = time.time()
    results_df = get_tesauro_query_results(st.session_state['icpc_search_input'])
    search_end = time.time()
    
    #results_df = results_df.groupby(['code'], as_index = False, sort=False).agg({'title': 'first', 'expression': ' | '.join})
    st.write(f"Busca entre {len(tesauro)} possibilidades realizada em {search_end-search_start:.2f} segundos.")
    st.write(f"Resultados encontrados para: **{st.session_state['icpc_search_input']}**")

    results_df = results_df.groupby(by='code').agg(' | '.join).reset_index()

    for row in results_df.to_dict('records'):
      code_criteria = get_code_criteria(row['code'][0:3])
      code_title = get_code_title(row['code'])
      with st.expander(f"__{row['code']} - {code_title}__"):
            st.write(f"_{row['expression']}_")
            st.write(f"**criterios de inclusão:** {code_criteria['inclusion criteria']}")
            st.write(f"**criterios de exclusão:** {code_criteria['exclusion criteria']}")

process = psutil.Process(os.getpid())
mem_info = process.memory_info()
print("Total memory usage:", mem_info.rss / (1024 ** 2), "MB")
