# Fastapi imports
from enum import Enum
from typing import Union
from fastapi import FastAPI
import json
from modulefinder import ModuleFinder

finder = ModuleFinder()
finder.run_script('bacon.py')

print('Loaded modules:')
for name, mod in finder.modules.items():
    print('%s: ' % name, end='')
    print(','.join(list(mod.globalnames.keys())[:3]))

# Imports for project
import pandas as pd
import numpy as np
from os.path import exists
import re
import datatable as dt
from tqdm import tqdm
from files import *
from functions import *

## This version will only load the models precomputed

# Dataframes with the data provided
print("Loading Models....")

genres  = dt.fread(file_genres).to_pandas().set_index('id')
info  = dt.fread(file_info).to_pandas().set_index('id')
youtube_urls = dt.fread(file_urls).to_pandas().set_index('id')

# Only to test
top_cosine_early_bert_blf_spectral_incp   = dt.fread(f_top_cosine_bert_blf_spectral_incp, header=True).to_pandas().set_index('index')
top_cosine_early_bert_blf_spectral_resnet = dt.fread(f_top_cosine_bert_blf_spectral_resnet, header=True).to_pandas().set_index('index')
top_cosine_early_bert_mfcc_bow_incp       = dt.fread(f_top_cosine_bert_mfcc_bow_incp, header=True).to_pandas().set_index('index')


class ModelName(str, Enum):
    modelA = "early_bert_blf_spectral_incp"
    modelB = "early_bert_blf_spectral_resnet"
    modelC = "early_bert_mfcc_bow_incp" 

class SimilarityFunction(Enum):
    cosine = "cosine"  # "Cosine Similarity"
    jaccard = "jaccard"  # "Jaccard Similarity"

topIdsFiles = {
    "cosine_early_bert_blf_spectral_incp" : top_cosine_early_bert_blf_spectral_incp,
    "cosine_early_bert_blf_spectral_resnet" : top_cosine_early_bert_blf_spectral_resnet,
    "cosine_early_bert_mfcc_bow_incp" : top_cosine_early_bert_mfcc_bow_incp,  
}

app = FastAPI()

@app.get("/")
def read_root():
    return {"API": "MMSR"}

@app.get("/query/")
async def getTopResults(artist: str, track: str, top: int, model: ModelName, simFunction: SimilarityFunction):
    print("\n\nGet Top results for \n\tArtist: ", artist, " \n\tTrack:",track, "\nUsing \n\tModel: ", model.name, "\n\tsimilarity function: ", simFunction.name, "\n\tTop: ", top)

    id_song = getSongIdByQuery(artist, track, info)
    if id_song == None:
        return { "error" : "No record found for this query"}
    print("\nId song:", id_song)

    file_id = simFunction.name + "_" + model
    print("file: ",file_id)
    if id_song in  topIdsFiles[file_id].index.values:
        print('\nQuery already in Top ids file for', file_id)
    else:
        print('\nNew song, calculating top 100 similar songs and saving to data')
    
    query_song = genres.loc[[id_song]].join(info, on="id").join(youtube_urls, on="id")

    top_n_ids = topIdsFiles[file_id].loc[id_song].values[:top]

    topVal = genres.loc[top_n_ids].join(info, on="id").join(youtube_urls, on="id")

    pk, mrrk, ndcgk = getMetrics(topIdsFiles[file_id].loc[[id_song]], top, genres)
    print("MAP@"+str(top), pk, "MRR@"+str(top), mrrk, "Mean NDCG@"+str(top), ndcgk, "\n\n")

    return { "song": json.loads(query_song.reset_index().to_json(orient='records')) , "top": json.loads(topVal.reset_index().to_json(orient='records'))  , "metrics" : { "MAP" : pk, "MRR" : mrrk, "NDCG" : ndcgk }  }