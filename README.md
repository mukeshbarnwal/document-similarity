# document-similarity

#Installing libraries
#cohere for embeddings, annoy for approx nearest neighbour search

!pip install cohere tqdm annoy


#Importing datasets
import cohere
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex


#API key
api_key='jQ8YO957htGlY4s62LN1M0EYpaf8UojIVAi9IPQG'

#Create and retrieve Choere API key from os.cohere.ai
co=cohere.Client(api_key)


