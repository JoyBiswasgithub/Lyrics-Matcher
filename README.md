# **Lyrics Matcher**
Lyrics Matcher is a machine learning-powered web application that helps users identify songs and artists based on a snippet of lyrics. It leverages Sentence Transformers for natural language processing and PyTorch for similarity search.


# Technologies Used
```
Python 

PyTorch 

Sentence Transformers 

Streamlit 

Pandas & NumPy 
```

#  Installation

**1️⃣ Prerequisites** 
Python 3.8+ installed 


**2️⃣ Clone the Repository**
```bash
git clone https://github.com/JoyBiswasgithub/Lyrics-Matcher.git```
cd lyrics-matcher
```

**3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

# Running the Application
```bash
streamlit run app.py
```


# 🌍 Live Demo

[🔗 Live Application](https://lyrics-matcher-app.streamlit.app/)


# 📂 Project Structure
```
lyrics-matcher/
│── data/
│   ├── song_info.csv
    ├── processed_songs.csv
│── sentence_transformer_model/
│── embeddings.pt
│── app.py  # Main application for similarity search and deployment
│── data_cleaning.ipynb  # Jupyter Notebook for data cleaning
│── embedding.ipynb  # Jupyter Notebook for converting data to embeddings
│── requirements.txt
│── README.md
```
