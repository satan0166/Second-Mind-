import streamlit as st
import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import torch
from transformers import pipeline

# Web Scraper
def scrape_web(query):
    sources = [
        f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
        f"https://www.sciencedaily.com/search/?keyword={query}",
        f"https://www.ncbi.nlm.nih.gov/pubmed/?term={query}"
    ]
    data = ""
    for url in sources:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            if paragraphs:
                data += " ".join(p.text for p in paragraphs) + " "
            else:
                st.warning(f"No <p> tags found on {url}")
        except Exception as e:
            st.warning(f"Error scraping {url}: {e}")
    return data.strip()

# Database Setup
def setup_database():
    conn = sqlite3.connect("second_mind.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY,
            query TEXT,
            output TEXT,
            score REAL,
            web_data TEXT
        )
    """)  # Fixed closing triple-quotes
    conn.commit()
    conn.close()

def store_interaction(query, output, score, web_data):
    conn = sqlite3.connect("second_mind.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO interactions (query, output, score, web_data)
        VALUES (?, ?, ?, ?)
    """, (query, output, score, web_data))
    conn.commit()
    conn.close()

# Agents
class GenerationAgent:
    def generate(self, query, web_data):
        return f"Hypothesis: Use {query} based on {web_data[:100]}..."

class ReflectionAgent:
    def check_coherence(self, hypothesis, web_data):
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([hypothesis, web_data])
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return similarity > 0.5

class RankingAgent:
    def score(self, hypothesis, web_data):
        # Calculate NDCG score
        gains = [1 if word in web_data else 0 for word in hypothesis.split()]
        dcg = sum([g / np.log2(i+2) for i, g in enumerate(gains)])
        idcg = sum([1 / np.log2(i+2) for i in range(len(gains))])
        ndcg_score = dcg / idcg if idcg > 0 else 0
        return round(ndcg_score * 10, 2)

class EvolutionAgent:
    def __init__(self):
        self.unmasker = pipeline("fill-mask", model="bert-base-uncased")

    def refine(self, hypothesis):
        words = hypothesis.split()
        refined_words = []
        for i, word in enumerate(words):
            masked_sentence = " ".join(words[:i] + ["[MASK]"] + words[i+1:])
            try:
                predictions = self.unmasker(masked_sentence)
                refined_word = predictions[0]['token_str']
            except Exception:
                refined_word = word
            refined_words.append(refined_word)
        return " ".join(refined_words)

# Supervisor
class Supervisor:
    def __init__(self):
        self.agents = {
            "generate": GenerationAgent(),
            "reflect": ReflectionAgent(),
            "rank": RankingAgent(),
            "evolve": EvolutionAgent()
        }

    def process_query(self, query):
        web_data = scrape_web(query)
        hypothesis = self.agents["generate"].generate(query, web_data)
        if not self.agents["reflect"].check_coherence(hypothesis, web_data):
            hypothesis = "Adjusted: " + hypothesis
        score = self.agents["rank"].score(hypothesis, web_data)
        refined_hypothesis = self.agents["evolve"].refine(hypothesis)
        store_interaction(query, refined_hypothesis, score, web_data)
        return refined_hypothesis, score

# Streamlit App
def main():
    st.title("The Second Mind: AI Agents for Iterative Learning")
    query = st.text_input("Enter your query (e.g., 'urban renewable energy'):")

    if query:
        setup_database()
        supervisor = Supervisor()
        scores = []

        for cycle in range(1, 4):
            st.subheader(f"Cycle {cycle}")
            hypothesis, score = supervisor.process_query(query)
            st.write(f"**Query:** {query}")
            st.write(f"**Hypothesis:** {hypothesis}")
            st.write(f"**Score:** {score}")
            scores.append(score)
            query = hypothesis

        st.subheader("Score Improvement Over Cycles")
        plt.plot(range(1, len(scores) + 1), scores, marker="o")
        plt.xlabel("Cycle")
        plt.ylabel("Score")
        st.pyplot(plt)

if __name__ == "__main__":
    main()
