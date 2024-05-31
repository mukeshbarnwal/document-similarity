# Document-similarity

# About
This code work is about comparing whether one document is similar to another in the database. 
If the document is similar to any document in the databse,  it will stop from uploading the document enabling storage efficiency, data maintainence, cost reduction and better tracking.

# Overview
**1. Getting ready with necesssary packages and libraries**
Cohere-> Conversion of text to word embeddings
Sklearn-> Compute cosine similarity
Pymupdf-> Read pdf and extract text for analysis
Fitz-> Read pdf and extract package from Pymupdf

**2. Conversion to word embeddings**
All stored documents are converted to contextual word embeddings which gives unique meaning to each of the documents.
The document to be uploaded is also converted in real-time to the contextual word embeddings. 

**3. Compare the embeddings of uploading documents with the existing ones**
Convert the text of uploading document into word embeddings and compare this word embedding with the embeddings of stored documents through the use of cosine similarity.
If it is greater than 99 percent then stop uploading.

# Installing libraries
#Cohere provides api for converting text into contextual word embeddings. The word embeddings capture the essence of meaning of text when read together.
!pip install cohere tqdm annoy #installing cohere library

# Importing packages
import cohere
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex


# API key
api_key='xyz...'
#Create and retrieve Choere API key from os.cohere.ai
co=cohere.Client(api_key)


# Distance between two documents using cohere

def get_document_embedding(paragraph):
    # Generate embeddings for the paragraph using Cohere
    response = co.embed(texts=[paragraph])
    return response.embeddings[0]


# Read pdf
pip install pymupdf

import fitz


#reading a file in Google Colab: source-> https://saturncloud.io/blog/how-to-read-a-file-from-drive-in-google-colab/
#mounting google drive
#this will grant permission to Colab to access the drive files
from google.colab import drive
drive.mount('/content/drive')
#drive is mounted and can be seen in the Files section


# Path to the PDF file
pdf_path = '/content/drive/MyDrive/Colab Notebooks/JD_Adobe.pdf'

# Extract text from pdf
def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Initialize an empty string to hold the extracted text
    extracted_text = ""

    # Loop through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        extracted_text += page.get_text()

    return extracted_text


def split_into_paragraphs(text):
    # Split the text into paragraphs
    paragraphs = text.split('\n\n')
    return paragraphs


# Extract text from the PDF
extracted_text = extract_text_from_pdf(pdf_path)

# Split the extracted text into paragraphs
paragraphs = split_into_paragraphs(extracted_text)


document=''.join(paragraphs)


#Get the embeddings for the paragraphs
embedding1 = get_document_embedding(document)


print(type(embedding1))
print(embedding1)


'''Convert string to list of paragraphs'''
paras=document.split('\n \n')


# Get the embeddings for the paragraphs
embedding1 = get_document_embedding(doc1)
embedding2 = get_document_embedding(doc1)


print(type(embedding1))
print(embedding1)
print(embedding2)


from scipy.spatial.distance import cosine

# Compute the cosine similarity
similarity = 1 - cosine(embedding1, embedding2)
print(f"Cosine similarity between the docs: {similarity}")


# Get the embeddings for the paragraphs
embedding1 = get_document_embedding(doc1)
embedding3 = get_document_embedding(doc2)


print(len(embedding1))
print(len(embedding3))
#standard length of vector embeddings is 4096


from scipy.spatial.distance import cosine

# Compute the cosine similarity
similarity = 1 - cosine(embedding1, embedding3)
print(f"Cosine similarity between the docs: {similarity}")


# Get the embeddings for the documents
embedding1 = get_document_embedding(doc1)
embedding2 = get_document_embedding(doc4)


from scipy.spatial.distance import cosine

# Compute the cosine similarity
similarity = 1 - cosine(embedding1, embedding2)
#removed last 2 pages from the original document
print(f"Cosine similarity between the docs: {similarity}")




from scipy.spatial.distance import cosine

# Compute the cosine similarity
similarity = 1 - cosine(embedding1, embedding2)
#removed only last from the original document
print(f"Cosine similarity between the docs: {similarity}")

#difference last 2 pages removed and only last page removed: 0.99, 0.996


