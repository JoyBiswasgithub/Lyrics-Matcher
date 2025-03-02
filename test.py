import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import random

# Load Model
model = SentenceTransformer('sentence_transformer_model')

# Load Embeddings and Song Data
embeddings = torch.load("embeddings.pt")
song_info = pd.read_csv("data/song_info.csv")

# Load Test Data
test_data = pd.read_csv("data/processed_songs.csv") 
data_low=random.randint(0,40000)
data_high = data_low + 100
test_data = test_data[data_low:data_high]  

# Function to find similar song
def find_similar_song(lyrics, top_k=10):
    input_embedding = model.encode(lyrics, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(input_embedding, embeddings)[0]
    top_results = torch.topk(similarities, k=top_k)
    
    results = []
    for idx in top_results.indices.tolist():
        results.append({
            "artist": song_info.iloc[idx]['artist'],
            "song": song_info.iloc[idx]['song'],
            "score": similarities[idx].item()
        })
    return results

# Evaluation Function
def evaluate_model(test_data, top_k=10):
    correct = 0
    
    for _, row in test_data.iterrows():
        lyrics = row['text']
        expected_song = row['song']
        expected_artist = row['artist']
        
        recommendations = find_similar_song(lyrics, top_k)
        
        for rec in recommendations:
            if rec['song'] == expected_song and rec['artist'] == expected_artist:
                correct += 1
                break  
    
    accuracy = (correct / len(test_data)) * 100
    return accuracy

# Run Evaluation
top_k = 1
accuracy = evaluate_model(test_data, top_k)

print(f"Top-{top_k} Accuracy: {accuracy:.2f}%")