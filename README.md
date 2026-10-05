# Facebook Friend Recommendation using Graph Mining

A link prediction and friend recommendation system that models social network connections as a directed graph. The project engineers graph-theoretic features (similarity measures, ranking algorithms, matrix factorization) and trains supervised machine learning models (Random Forest, XGBoost) to predict missing links between users, complemented by an interactive Streamlit dashboard and Neo4j graph database integration.

---

## 🌟 Key Features

- **Exploratory Data Analysis (EDA):** In-depth graph structural analysis (in-degree/out-degree distribution, connected components, follower/following ratios).
- **Advanced Graph Featurization:**
  - **Similarity Indices:** Jaccard Distance, Cosine Distance for followers and followees.
  - **Ranking & Centrality:** PageRank, Katz Centrality, HITS (Hubs and Authorities).
  - **Shortest Path & Community Detection:** Weakly connected components, shortest path calculation (excluding direct edge).
  - **Matrix Factorization (SVD):** U and V latent embeddings derived from adjacency matrix decomposition.
  - **Preferential Attachment & Adamic/Adar Index:** For topological neighbor overlap.
- **Machine Learning Models:**
  - Random Forest Classifier
  - XGBoost Classifier (with Random Search hyperparameter tuning)
  - Evaluation via F1-score, Precision, Recall, and Confusion Matrices.
- **Interactive Streamlit Web Dashboard (`frontend.py`):**
  - Real-time link prediction between any two user IDs.
  - Interactive graph visualizations (Plotly & NetworkX).
  - Feature importance breakdown and model metrics comparison.
- **Neo4j Graph Database Integration (`neo4j_db.py`, `load_to_neo4j.py`):**
  - AuraDB / local Neo4j cloud connection for fast graph traversals and Cypher queries.

---

## 📁 Repository Structure

```
├── .streamlit/
│   └── secrets.toml.example        # Template for Neo4j Aura credentials
├── FB_EDA - 1.ipynb                # Graph EDA & baseline statistics
├── FB_featurization.ipynb          # Similarity measures & centrality features
├── FB-FE-2.ipynb                   # Feature engineering phase 2
├── FB_Models - 3.ipynb             # Model training (RF & XGBoost)
├── Finale.ipynb                    # End-to-end pipeline notebook
├── frontend.py                     # Streamlit web application
├── neo4j_db.py                     # Neo4j driver and Cypher query helpers
├── load_to_neo4j.py                # Batch loader for graph data into Neo4j
├── compress_model.py               # Utility to compress model artifacts
├── requirements.txt                # Python package dependencies
├── .gitignore                      # Git ignore rules for data and secrets
└── README.md                       # Project documentation
```

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone -b develop https://github.com/Dexter3110/Facebook-Recommendations.git
cd Facebook-Recommendations
```

### 2. Set up virtual environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure Neo4j AuraDB credentials
Copy the example file:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Update `.streamlit/secrets.toml` with your Neo4j instance URI, username, and password.

### 5. Launch the Streamlit dashboard
```bash
streamlit run frontend.py
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Graph & Math:** NetworkX, SciPy, NumPy, Pandas
- **Machine Learning:** Scikit-Learn, XGBoost
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Graph Database:** Neo4j (Cypher)
- **Web Interface:** Streamlit
