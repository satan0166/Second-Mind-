import os
import asyncio
import requests
import json
import aiohttp
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from bs4 import BeautifulSoup
from transformers import pipeline

# Load API Key from environment variable (Set it in VS Code using os.environ)
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Initialize Firebase (Check to prevent re-initialization error)
json_path = os.path.join(os.getcwd(), "second-mind-17e47-firebase-adminsdk-fbsvc-4d3c655e7b.json")
if not firebase_admin._apps:
    cred = credentials.Certificate(json_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ----------------- Async Web Scraping -----------------
async def async_scrape_web(query):
    """Scrapes search results using SerpAPI and additional sources asynchronously."""
    web_data = ""

    async def fetch_url(url):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as response:
                    if response.status == 200:
                        text = await response.text()
                        soup = BeautifulSoup(text, "html.parser")
                        paragraphs = soup.find_all("p")
                        return " ".join(p.text for p in paragraphs) if paragraphs else ""
        except Exception as e:
            return f"Error fetching {url}: {e}"

    # SerpAPI search request
    search_url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    try:
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            organic_results = data.get("organic_results", [])
            web_data = " ".join([result.get("snippet", "") for result in organic_results])
    except Exception as e:
        print(f"Error fetching SerpAPI results: {e}")

    # Additional sources
    additional_sources = [
        f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
        f"https://www.sciencedaily.com/search/?keyword={query}",
        f"https://www.ncbi.nlm.nih.gov/pubmed/?term={query}",
        f"https://www.researchgate.net/search/publication?q={query}"
    ]
    
    fetched_data = await asyncio.gather(*(fetch_url(url) for url in additional_sources))
    web_data += " ".join(fetched_data)

    return web_data.strip()

# ----------------- AI Agents -----------------
class GenerationAgent:
    def __init__(self):
        self.generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct", device=-1)

    def generate(self, query, web_data):
        prompt = f"Generate a well-reasoned hypothesis based on the following information:\n\nQuery: {query}\n\nWeb Data: {web_data[:1000]}\n\nHypothesis:"
        result = self.generator(prompt, max_new_tokens=150, num_return_sequences=1, do_sample=True)
        return result[0]['generated_text']

class FastGenerationAgent:
    def __init__(self):
        self.generator = pipeline("text-generation", model="distilgpt2", device=-1)

    def generate(self, query, web_data):
        prompt = f"Generate a brief, fast hypothesis based on:\n\nQuery: {query}\n\nWeb Data: {web_data[:500]}\n\nHypothesis:"
        result = self.generator(prompt, max_new_tokens=100, num_return_sequences=1, do_sample=True)
        return result[0]['generated_text']

class ReflectionAgent:
    def check_coherence(self, hypothesis, query):
        return query.lower() in hypothesis.lower()

class RankingAgent:
    def score(self, hypothesis, web_data, query):
        relevance_score = web_data.lower().count(query.lower()) * 0.5
        length_score = min(len(hypothesis) * 0.1, 10)
        return int(relevance_score + length_score)

import torch
from transformers import pipeline

class EvolutionAgent:
    def __init__(self):
        # Load BERT-based text infilling model
        self.unmasker = pipeline("fill-mask", model="bert-base-uncased")

    def refine(self, hypothesis, web_data):
        words = hypothesis.split()
        refined_words = []

        for i, word in enumerate(words):
            # Mask the word and predict replacements
            masked_sentence = " ".join(words[:i] + ["[MASK]"] + words[i+1:])
            try:
                predictions = self.unmasker(masked_sentence)
                refined_word = predictions[0]['token_str']  # Top prediction
            except Exception:
                refined_word = word  # Keep original word in case of an error
            
            refined_words.append(refined_word)

        return " ".join(refined_words)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ProximityAgent:
    def __init__(self):
        # Initialize Firebase (Ensure you have the service account JSON)
        if not firebase_admin._apps:  # Prevent re-initialization error
            cred = credentials.Certificate("path/to/your/serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()
    
    def recall_interaction(self, query, top_n=3):
        # Fetch past interactions from Firestore
        interactions_ref = self.db.collection("interactions").stream()
        past_interactions = [(doc.to_dict()["query"], doc.to_dict()["output"]) for doc in interactions_ref]

        if not past_interactions:
            return ["No past interactions found."]

        # Prepare text data for similarity comparison
        texts = [query] + [f"{q} {o}" for q, o in past_interactions]

        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Compute cosine similarity with the current query
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Get top N most similar interactions
        top_indices = similarities.argsort()[-top_n:][::-1]
        top_matches = [past_interactions[i] for i in top_indices]

        return [f"Query: {q} | Output: {o}" for q, o in top_matches]
import re

class MetaReviewAgent:
    def __init__(self, exec_time_threshold=5.0):
        """
        Initializes the MetaReviewAgent with performance thresholds.
        
        Parameters:
        exec_time_threshold (float): Time in seconds above which performance optimization is suggested.
        """
        self.exec_time_threshold = exec_time_threshold
        self.error_patterns = {
            "Scraper": re.compile(r"Error scraping", re.IGNORECASE),
            "Database": re.compile(r"database|Firebase", re.IGNORECASE),
            "API": re.compile(r"API|request failed", re.IGNORECASE),
            "Agent": re.compile(r"Exception in (Generation|Reflection|Ranking|Evolution|Proximity)Agent", re.IGNORECASE)
        }
        self.exec_time_pattern = re.compile(r"Execution time: (\d+\.\d+)")

    def evaluate(self, logs):
        """
        Evaluates logs to detect errors, performance issues, and system efficiency.

        Parameters:
        logs (list[str]): A list of log messages.

        Returns:
        str: A structured meta-review summary.
        """
        error_count = {key: 0 for key in self.error_patterns}
        total_errors = 0
        execution_times = []

        # Process logs efficiently
        for log in logs:
            for error_type, pattern in self.error_patterns.items():
                if pattern.search(log):
                    error_count[error_type] += 1
                    total_errors += 1
            
            exec_time_match = self.exec_time_pattern.search(log)
            if exec_time_match:
                execution_times.append(float(exec_time_match.group(1)))

        # Generate review summary
        review = []
        if total_errors:
            review.append(f"⚠️ **{total_errors} errors detected. Improve error handling.**")
            review.extend([f"- {count} {etype}-related errors found." for etype, count in error_count.items() if count > 0])

        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            if avg_time > self.exec_time_threshold:
                review.append(f"⏳ **Execution time is high ({avg_time:.2f}s). Optimize performance.**")

        return "\n".join(review) if review else "✅ **Process is efficient. No major issues detected.**"


# Example usage
meta_agent = MetaReviewAgent()
logs = [
    "Execution time: 6.2",
    "Error scraping Wikipedia: Timeout",
    "Database connection failed",
    "Execution time: 4.1"
]
review = meta_agent.evaluate(logs)
print(review)


# ----------------- Supervisor Class -----------------
class Supervisor:
    def __init__(self):
        self.agents = {
            "generate": GenerationAgent(),
            "fast_generate": FastGenerationAgent(),
            "reflect": ReflectionAgent(),
            "rank": RankingAgent(),
            "evolve": EvolutionAgent(),
            "proximity": ProximityAgent(),
            "meta": MetaReviewAgent()
        }
    
    async def process_query(self, query):
        web_data = await async_scrape_web(query)
        hypothesis = self.agents["fast_generate"].generate(query, web_data)
        
        if not self.agents["reflect"].check_coherence(hypothesis, query):
            hypothesis = "Adjusted: " + hypothesis
        
        score = self.agents["rank"].score(hypothesis, web_data, query)
        refined_hypothesis = self.agents["evolve"].refine(hypothesis)
        
        db.collection("interactions").document().set({
            "query": query,
            "hypothesis": refined_hypothesis,
            "score": score,
            "web_data": web_data
        })

        logs = f"Processed {query} with score {score}"
        return refined_hypothesis, score, self.agents["meta"].evaluate(logs)

# ----------------- Streamlit App -----------------
def main():
    st.title("AI Hypothesis Generator")

    query = st.text_input("Enter a topic:")
    if st.button("Generate Hypothesis"):
        if not query:
            st.warning("Please enter a topic.")
        else:
            supervisor = Supervisor()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            refined_hypothesis, score, meta_evaluation = loop.run_until_complete(supervisor.process_query(query))

            st.subheader("Generated Hypothesis")
            st.write(refined_hypothesis)
            st.write(f"Relevance Score: {score}")
            st.write(f"Meta Review: {meta_evaluation}")

if __name__ == "__main__":
    main()
