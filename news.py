import os
from dotenv import load_dotenv
import streamlit as st
from newsapi import NewsApiClient
import cohere
import torch
import numpy as np

load_dotenv(".env")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
cohere_client = cohere.Client(api_key=COHERE_API_KEY)

# Embed text using Cohere
torchfy = lambda x: torch.tensor(x, dtype=torch.float32)

st.set_page_config(layout="wide")
st.title("News Article Search and Summarization")

query = st.text_input("Enter your search query:", "")

def normalize_embeddings(embeddings):
    norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    return embeddings / norm

def get_similarity(target_embedding, candidate_embeddings, top_k=10):
    target_tensor = torchfy(target_embedding).reshape(1, -1)
    candidate_tensor = torchfy(candidate_embeddings)

    target_tensor = normalize_embeddings(target_tensor)
    candidate_tensor = normalize_embeddings(candidate_tensor)

    cos_scores = torch.mm(target_tensor, candidate_tensor.T).squeeze(0)
    top_scores, top_indices = torch.topk(cos_scores, k=top_k)
    return top_scores.tolist(), top_indices.tolist()

if st.button("Search") or query:
    if query:
        news_articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=100)
        if news_articles['status'] == 'ok':
            articles = news_articles['articles']
            article_descriptions = [article['description'] for article in articles]

            query_embedding = cohere_client.embed(model="small", texts=[query]).embeddings[0]
            description_embeddings = cohere_client.embed(model="small", texts=article_descriptions).embeddings

            scores, indices = get_similarity(query_embedding, description_embeddings)

            st.markdown(f"### Found {len(articles)} articles matching '{query}'")
            for rank, idx in enumerate(indices[:10], start=1):
                article = articles[idx]
                st.markdown(f"#### {rank}. [{article['title']}]({article['url']}) ({article['source']['name']})")
                st.write(article['description'])
                st.write(f"**Similarity Score:** {scores[rank-1]:.4f}")
                st.markdown("---")
        else:
            st.write("No articles found. Please try again.")
