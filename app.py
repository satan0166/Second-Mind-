import os
import streamlit as st
import requests
import json
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import firebase_admin
from firebase_admin import credentials, firestore
from transformers import pipeline
import matplotlib.pyplot as plt

# Load API Key from environment variable
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Initialize Firebase
json_path = "C:\\Users\\ANSHUMAN\\Downloads\\second-mind-17e47-firebase-adminsdk-fbsvc-4d3c655e7b.json"
if not firebase_admin._apps:
    cred = credentials.Certificate(json_path)  # Ensure correct path
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Asynchronous Web Scraping Function (Using SerpAPI + Additional Sources)
async def async_scrape_web(query):
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

    # Fetch search results
    search_url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    try:
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            organic_results = data.get("organic_results", [])
            web_data = " ".join([result.get("snippet", "") for result in organic_results])
    except Exception as e:
        st.warning(f"Error fetching SerpAPI results: {e}")

    # Fetch additional sources asynchronously
    additional_sources = [
        f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
        f"https://www.sciencedaily.com/search/?keyword={query}",
        f"https://www.ncbi.nlm.nih.gov/pubmed/?term={query}",
        f"https://www.researchgate.net/search/publication?q={query}"
    ]
    
    fetched_data = await asyncio.gather(*(fetch_url(url) for url in additional_sources))
    web_data += " ".join(fetched_data)

    return web_data.strip()

# AI Agents
class GenerationAgent:
    def __init__(self):
        self.generator = pipeline("text-generation", model="facebook/opt-1.3b", device=-1)

    def generate(self, query, web_data):
        prompt = f"Generate a well-reasoned hypothesis based on the following information:\n\nQuery: {query}\n\nWeb Data: {web_data[:1000]}\n\nHypothesis:"
        result = self.generator(prompt, max_new_tokens=150, num_return_sequences=1, do_sample=True)
        return result[0]['generated_text']

class ReflectionAgent:
    def check_coherence(self, hypothesis, query):
        return query.lower() in hypothesis.lower()

class RankingAgent:
    def score(self, hypothesis, web_data, query):
        relevance_score = web_data.lower().count(query.lower()) * 0.5
        length_score = min(len(hypothesis) * 0.1, 10)
        return int(relevance_score + length_score)

class EvolutionAgent:
    def refine(self, hypothesis):
        return hypothesis.replace("AI", "Advanced AI") if "AI" in hypothesis.lower() else hypothesis

class ProximityAgent:
    def find_links(self, query):
        return [doc.to_dict() for doc in db.collection("interactions").where("query", "==", query).stream()]

class MetaReviewAgent:
    def evaluate(self, logs):
        return "Improve error handling and optimize web scraping." if "error" in logs else "Process is efficient."

class Supervisor:
    def __init__(self):
        self.agents = {
            "generate": GenerationAgent(),
            "reflect": ReflectionAgent(),
            "rank": RankingAgent(),
            "evolve": EvolutionAgent(),
            "proximity": ProximityAgent(),
            "meta": MetaReviewAgent()
        }
    
    async def process_query(self, query):
        web_data = await async_scrape_web(query)
        hypothesis = self.agents["generate"].generate(query, web_data)
        
        if not self.agents["reflect"].check_coherence(hypothesis, query):
            hypothesis = "Adjusted: " + hypothesis
        
        score = self.agents["rank"].score(hypothesis, web_data, query)
        refined_hypothesis = self.agents["evolve"].refine(hypothesis)
        
        # Store interaction in Firebase
        db.collection("interactions").document().set({
            "query": query,
            "hypothesis": refined_hypothesis,
            "score": score,
            "web_data": web_data
        })

        logs = f"Processed {query} with score {score}"
        return refined_hypothesis, score, self.agents["meta"].evaluate(logs)

# Streamlit UI
def main():
    st.set_page_config(page_title="The Second Mind", layout="wide")
    st.title("🤖 The Second Mind: AI Agents for Iterative Learning")
    query = st.text_input("🔍 Enter your query (e.g., 'Urban Renewable Energy'):")

    if query:
        supervisor = Supervisor()
        scores, interactions = [], []
        
        with st.spinner("🔄 Processing..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            for cycle in range(1, 4):
                hypothesis, score, evaluation = loop.run_until_complete(supervisor.process_query(query))
                interactions.append({"cycle": cycle, "query": query, "hypothesis": hypothesis, "score": score})
                
                st.subheader(f"🌀 Cycle {cycle}")
                st.write(f"**Query:** {query}")
                st.write(f"**Hypothesis:** {hypothesis}")
                st.write(f"**Score:** {score}/10")
                st.write(f"📌 **Evaluation:** {evaluation}")
                
                scores.append(score)
                query = hypothesis  # Iterative improvement
        
        # Store session logs in Firestore
        db.collection("session_logs").document().set({"interactions": interactions})

        # Score Improvement Chart
        st.subheader("📊 Score Improvement Over Cycles")
        fig, ax = plt.subplots()
        ax.plot(range(1, len(scores) + 1), scores, marker="o", linestyle="-", color="b")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Score")
        ax.set_title("Score Progression")
        ax.grid(True)
        st.pyplot(fig)

        # Stored Interactions
        st.subheader("📁 Stored Interactions")
        st.json(interactions)

if __name__ == "__main__":
    main()
