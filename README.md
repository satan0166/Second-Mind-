# The Second Mind: AI Agents for Iterative Learning

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD43B?style=for-the-badge&logo=huggingface&logoColor=black)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)

---

## **Overview**

"The Second Mind" is a Streamlit-based web application that simulates a system of AI agents designed to mimic human learning. The app processes user input, generates hypotheses, refines them iteratively, and improves over multiple cycles. It incorporates real-time web scraping, text generation, and feedback-driven improvement to achieve its goals.

The system consists of six specialized agents:
1. **Generation Agent**: Creates initial hypotheses based on user input and web data.
2. **Reflection Agent**: Checks the coherence of hypotheses with web data.
3. **Ranking Agent**: Scores hypotheses based on relevance and length.
4. **Evolution Agent**: Refines hypotheses iteratively.
5. **Proximity Agent**: Links to past interactions for context.
6. **Meta-Review Agent**: Evaluates the process and suggests improvements.

---

## **Features**

- **Real-Time Web Scraping**: Fetches data from multiple sources (e.g., Wikipedia, ScienceDaily, PubMed) to inform hypotheses.
- **Text Generation**: Uses Hugging Face's `transformers` library (GPT-2) to generate hypotheses.
- **Iterative Improvement**: Runs 2–3 cycles to refine hypotheses and improve scores.
- **Data Visualization**: Plots the improvement in scores over cycles using Matplotlib.
- **Persistent Storage**: Stores past interactions in an SQLite database for recall and context.

---

## **Tech Stack**

- **Frontend**: Streamlit
- **Backend**: Python
- **AI & NLP**: Hugging Face Transformers, PyTorch, Scikit-learn
- **Web Scraping**: BeautifulSoup, Requests
- **Database**: SQLite
- **Data Visualization**: Matplotlib
- **Deployment**: Streamlit Sharing, Heroku, Docker (optional)

---

## **Getting Started**

### **Prerequisites**

- Python 3.8 or higher
- Pip (Python package manager)

### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/the-second-mind.git
   cd the-second-mind
