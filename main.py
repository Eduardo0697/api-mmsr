# Fastapi imports
from enum import Enum
from typing import Union
from fastapi import FastAPI
import json
from fastapi.middleware.cors import CORSMiddleware

# Imports for project
import pandas as pd
import numpy as np
import datatable as dt
from functions import *
from files import *

## This version will only load the models precomputed

# Dataframes with the data provided
print("Loading Models....")

genres  = dt.fread(file_genres).to_pandas().set_index('id')
info  = dt.fread(file_info).to_pandas().set_index('id')
youtube_urls = dt.fread(file_urls).to_pandas().set_index('id')
id_numbers  = pd.read_csv('./data/relation_id_number.csv').set_index('idNumber')


dtypes = {'index': 'str'} | dict(zip(range(100), ['int32' for i in range(100)]))

# Only to test
top_cosine_bert_mfcc_bow_incp = pd.read_csv('./data/model_selected.csv', dtype=dtypes).set_index("index")

class ModelName(str, Enum):
    model = "model"


topIdsFiles = {
    "model" : top_cosine_bert_mfcc_bow_incp,
}

# import psutil
 
# Getting % usage of virtual_memory ( 3rd field)
# print('RAM memory % used:', psutil.Process().memory_info().rss / (1024 * 1024))


app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "https://mmsr-app.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"API": "MMSR"}

@app.get("/query/")
async def getTopResults(artist: str, track: str, top: int, model: ModelName):
    print("\n\nGet Top results for \n\tArtist: ", artist, " \n\tTrack:",track, "\nUsing \n\tModel: ", model.name, "\n\tTop: ", top)

    id_song = getSongIdByQuery(artist, track, info)
    if id_song == None:
        return { "error" : "No record found for this query"}
    print("\nId song:", id_song)

    file_id = model.name
    print("file: ",file_id)
    if id_song in  topIdsFiles[file_id].index.values:
        print('\nQuery already in Top ids file for', file_id)
    else:
        print('\nNew song, calculating top 100 similar songs and saving to data')
    
    query_song = genres.loc[[id_song]].join(info, on="id").join(youtube_urls, on="id")

    top_n_ids = topIdsFiles[file_id].loc[id_song].values[:top]
    # print("Topids",top_n_ids)
    ids = id_numbers.loc[top_n_ids].values.flatten()
    # print("ids", ids)
    topVal = genres.loc[ids].join(info, on="id").join(youtube_urls, on="id")
    
    # print(topIdsFiles[file_id].loc[[id_song]].apply(lambda s,ids : [ids.loc[x].values for x in s], raw=True, axis=1, ids=id_numbers))

    # pk, mrrk, ndcgk = getMetrics(topIdsFiles[file_id].loc[[id_song]], top, genres)
    # print("MAP@"+str(top), pk, "MRR@"+str(top), mrrk, "Mean NDCG@"+str(top), ndcgk, "\n\n")

    return { 
        "song": json.loads(query_song.reset_index().to_json(orient='records')) , 
        "top": json.loads(topVal.reset_index().to_json(orient='records'))  , 
        # "metrics" : { "MAP" : pk, "MRR" : mrrk, "NDCG" : ndcgk } 
         }