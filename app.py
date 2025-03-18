import streamlit as st
import requests
from bs4 import BeautifulSoup
import sqlite3
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

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
            # Extract text from all <p> tags
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
            score INTEGER,
            web_data TEXT
        )
    """)
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

def recall_interaction(query):
    conn = sqlite3.connect("second_mind.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT output, score, web_data FROM interactions
        WHERE query = ?
    """, (query,))
    result = cursor.fetchall()
    conn.close()
    return result

# Agents
class GenerationAgent:
    def generate(self, query, web_data):
        # Simple rule-based hypothesis generation
        return f"Hypothesis: Use {query} based on {web_data[:100]}..."

class ReflectionAgent:
    def check_coherence(self, hypothesis, web_data, query):
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([hypothesis, web_data])
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return similarity > 0.5

class RankingAgent:
    def score(self, hypothesis, web_data, query):
        # Simple rule-based scoring
        score = 0
        score += len(hypothesis) * 0.1  # Longer hypotheses get higher scores
        score += web_data.count(query) * 0.5  # More mentions of the query in web data
        return min(int(score), 10)  # Cap score at 10

class EvolutionAgent:
    def refine(self, hypothesis, web_data):
        if "solar" in hypothesis.lower():
            return hypothesis.replace("solar", "advanced solar")
        return hypothesis

class ProximityAgent:
    def find_links(self, query):
        return recall_interaction(query)

class MetaReviewAgent:
    def evaluate(self, logs):
        if "error" in logs:
            return "Improve error handling."
        return "Process is efficient."

# Supervisor
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

    def process_query(self, query):
        web_data = scrape_web(query)
        hypothesis = self.agents["generate"].generate(query, web_data)
        if not self.agents["reflect"].check_coherence(hypothesis, web_data, query):
            hypothesis = "Adjusted: " + hypothesis
        score = self.agents["rank"].score(hypothesis, web_data, query)
        refined_hypothesis = self.agents["evolve"].refine(hypothesis, web_data)
        store_interaction(query, refined_hypothesis, score, web_data)
        logs = f"Processed {query} with score {score}"
        evaluation = self.agents["meta"].evaluate(logs)
        return refined_hypothesis, score, evaluation

# Streamlit App
def main():
    st.title("The Second Mind: AI Agents for Iterative Learning")
    st.write("This app uses AI agents to process user input, refine hypotheses, and improve over multiple cycles.")

    # User input
    query = st.text_input("Enter your query (e.g., 'urban renewable energy'):")

    if query:
        setup_database()
        supervisor = Supervisor()
        scores = []

        # Run 2-3 cycles
        for cycle in range(1, 4):  # Run 3 cycles
            st.subheader(f"Cycle {cycle}")
            hypothesis, score, evaluation = supervisor.process_query(query)
            st.write(f"**Query:** {query}")
            st.write(f"**Hypothesis:** {hypothesis}")
            st.write(f"**Score:** {score}")
            st.write(f"**Evaluation:** {evaluation}")
            scores.append(score)
            
            # Update query for next cycle
            query = hypothesis

        # Visualize scores
        st.subheader("Score Improvement Over Cycles")
        plt.plot(range(1, len(scores) + 1), scores, marker="o")
        plt.xlabel("Cycle")
        plt.ylabel("Score")
        st.pyplot(plt)

# Run the app
if __name__ == "__main__":
    main()


#text versctorization reomve it '
#firebase integration 
#pandas remove no use 
#