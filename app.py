import streamlit as st
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

#load Model
model=SentenceTransformer('sentence_transformer_model')
# Load Embedding and Song Details
embeddings = torch.load("embeddings.pt")
song_info = pd.read_csv("data/song_info.csv")

# Find similar song
def find_similar_song(lyrics, top_k=1):
    input_embedding = model.encode(lyrics, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(input_embedding, embeddings)[0]
    top_results = torch.topk(similarities, k=top_k)

    results = []
    for idx in top_results.indices.tolist():  # Convert tensor indices to list
        results.append({
            "artist": song_info.iloc[idx]['artist'],
            "song": song_info.iloc[idx]['song'],
            "score": similarities[idx].item()
        })

    return results

# Streamlit UI
st.title("🎵 Song Finder")
st.write("Find songs based on a snippet of lyrics!")

lyrics = st.text_input("Enter your lyrics snippet:")


# Handle search on button click
if st.button("Find Song"):
    if lyrics:  
        result = find_similar_song(lyrics)
        if result[0]['score'] < .4:
            st.error("No matching song found. Try a different lyrics!")
        else:
            st.subheader("Most Recommended")
            st.write(f"Song Name: {result[0]['song']}  \nArtist Name: {result[0]['artist']}")
           
    else:
        st.warning("Please enter some lyrics.")


