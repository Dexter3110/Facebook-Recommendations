import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gzip
import warnings
warnings.filterwarnings('ignore')

# Neo4j connectivity (optional – app keeps working if unavailable)
try:
    import neo4j_db
    _NEO4J_AVAILABLE = True
except Exception:
    _NEO4J_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Facebook Friend Recommendation System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling - ALL TEXT WHITE
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1877F2 !important;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 2rem;
        color: #42A5F5 !important;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E3F2FD;
        padding-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1877F2 !important;
        margin: 2rem 0 1rem 0;
        font-weight: bold;
    }
    
    /* Info boxes with dark text on light background */
    .info-box {
        background-color: #E3F2FD !important;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1877F2;
        margin: 1rem 0;
    }
    .info-box h2, .info-box h3, .info-box h4, .info-box p, .info-box li, .info-box b, .info-box strong {
        color: #000000 !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Success boxes with dark text on light background */
    .success-box {
        background-color: #E8F5E8 !important;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .success-box h3, .success-box h4, .success-box p, .success-box li, .success-box b, .success-box strong {
        color: #000000 !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Warning boxes with dark text on light background */
    .warning-box {
        background-color: #FFF3E0 !important;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .warning-box h3, .warning-box h4, .warning-box p, .warning-box li, .warning-box b, .warning-box strong {
        color: #000000 !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Metric cards with dark text on white background */
    .metric-card {
        background: white !important;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
        border: 1px solid #E0E0E0;
    }
    .metric-card h3, .metric-card p, .metric-card b, .metric-card strong {
        color: #000000 !important;
        font-size: 1rem;
        margin: 0.3rem 0;
    }
    
    /* Friend cards */
    .friend-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .friend-card * {
        color: white !important;
    }
    
    /* Node badges */
    .node-badge {
        background-color: #1877F2;
        color: white !important;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    /* Divider */
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #1877F2, #42A5F5, #1877F2);
        margin: 3rem 0;
        border-radius: 2px;
    }
    
    /* General white text for main content */
    .stApp, .main, .block-container {
        color: #FFFFFF !important;
    }
    
   .stMarkdown p:not(.info-box *):not(.success-box *):not(.warning-box *):not(.metric-card *), 
.stMarkdown li:not(.info-box *):not(.success-box *):not(.warning-box *):not(.metric-card *), 
.stMarkdown span:not(.info-box *):not(.success-box *):not(.warning-box *):not(.metric-card *), 
.stMarkdown div:not(.info-box *):not(.success-box *):not(.warning-box *):not(.metric-card *) {
    color: #FFFFFF !important;
}
    
    
    
    /* Remove gap between header and input in columns */
div[data-testid="column"] .stMarkdown + div {
    margin-top: -1rem !important;
}

/* Specific targeting for number input after header */
div[data-testid="column"] h3 + div[data-testid="stNumberInput"] {
    margin-top: -1.5rem !important;
}

/* Remove default spacing from streamlit elements in columns */
div[data-testid="column"] > div {
    margin-bottom: 0.5rem !important;
}
    
    /* Code elements */
    .stMarkdown code {
        background-color: #1E88E5 !important;
        color: #FFFFFF !important;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }
    
    /* Tables */
    .stDataFrame, .stTable {
        color: #000000 !important;
    }
    
    /* Input labels */
    .stNumberInput label, .stSlider label {
        color: #FFFFFF !important;
    }
    
    /* Metrics */
    .stMetric {
        color: #FFFFFF !important;
    }
    
    /* Buttons */
    .stButton button {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_data
def load_data():
    """Load all necessary data and models"""
    try:
        # Load top 30 nodes data
        top_30_df = pd.read_csv('top_30_warm_start_analysis_2plus_friends.csv')
        
        # Load test data
        df_test = pd.read_csv('featured_test_data.csv.gz', compression='gzip')
        
        return top_30_df, df_test
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_resource
def load_model():
    """Load the compressed trained Random Forest model"""
    try:
        # Load compressed model
        with gzip.open('best_random_forest_model_compressed.pkl.gz', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def recommend_friends(source_node_id, df_test, model, top_k=5):
    """
    Recommend friends for a given source node using the trained model
    """
    # Filter potential connections for the source node
    potential_connections = df_test[df_test['source_node'] == source_node_id].copy()
    
    if potential_connections.empty:
        return {
            'status': 'No potential connections found',
            'recommendations': [],
            'total_candidates': 0
        }
    
    # Prepare features
    X_potential = potential_connections.drop(['source_node', 'destination_node', 'indicator_link'], axis=1)
    
    # Get prediction probabilities
    probabilities = model.predict_proba(X_potential)[:, 1]
    
    # Get destination nodes and actual labels
    destinations = potential_connections['destination_node'].values
    actual_labels = potential_connections['indicator_link'].values
    
    # Create recommendation list
    recommendations = []
    for i, dest_node in enumerate(destinations):
        recommendations.append({
            'destination_node': int(dest_node),
            'friendship_probability': float(probabilities[i]),
            'actual_connection': int(actual_labels[i]),
            'confidence': 'High' if probabilities[i] > 0.7 else 'Medium' if probabilities[i] > 0.5 else 'Low'
        })
    
    # Sort by probability
    recommendations.sort(key=lambda x: x['friendship_probability'], reverse=True)
    
    return {
        'status': 'Success',
        'recommendations': recommendations[:top_k],
        'total_candidates': len(recommendations),
        'actual_friends': sum(1 for r in recommendations if r['actual_connection'] == 1)
    }

def display_friend_icon(friend_id, probability, confidence, actual_connection):
    """Display a friend recommendation with icon"""
    confidence_colors = {
        'High': '#4CAF50',
        'Medium': '#FF9800', 
        'Low': '#F44336'
    }
    
    status_emoji = "✅" if actual_connection == 1 else "❌"
    
    st.markdown(f"""
    <div class="friend-card">
        <div style="font-size: 2rem; color: white !important;">👤</div>
        <div style="font-size: 1.2rem; font-weight: bold; color: white !important;">Friend ID: {friend_id}</div>
        <div style="font-size: 1rem; margin: 0.5rem 0; color: white !important;">
            Probability: <b style="color: white !important;">{probability:.3f}</b>
        </div>
        <div style="background-color: {confidence_colors[confidence]}; 
                    padding: 0.3rem; border-radius: 15px; margin: 0.5rem 0; color: white !important;">
            {confidence} Confidence
        </div>
        <div style="font-size: 1.1rem; color: white !important;">
            {status_emoji} {"Real Friend" if actual_connection == 1 else "Not Connected"}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main App
def main():
    # ── Neo4j sidebar status indicator ──────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🗄️ Database")
        if _NEO4J_AVAILABLE:
            connected = neo4j_db.is_connected()
        else:
            connected = False

        if connected:
            st.markdown(
                "<span style='color:#4CAF50; font-weight:bold;'>● Neo4j: connected</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='color:#F44336; font-weight:bold;'>● Neo4j: not connected</span>",
                unsafe_allow_html=True,
            )
        st.markdown("---")

    # Load data and model
    top_30_df, df_test = load_data()
    model = load_model()

    if top_30_df is None or df_test is None or model is None:
        st.error("Failed to load required data or model. Please check file paths.")
        return
    
    # Initialize session state for user input
    if 'selected_node' not in st.session_state:
        st.session_state.selected_node = 1492489
    
    # Header
    st.markdown('<div class="main-header">👥 Facebook Friend Recommendation System</div>', 
                unsafe_allow_html=True)
    
    # Welcome Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="info-box">
            <h2>🎯 Welcome to the Facebook Friend Recommendation System!</h2>
            <p>This system uses advanced machine learning algorithms to predict and recommend potential friends 
            based on graph mining techniques and social network analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown('<div class="sub-header">📊 System Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #1877F2;">🎯 Model</h3>
            <p><b>Random Forest</b></p>
            <p>121 estimators</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #42A5F5;">📈 F1 Score</h3>
            <p><b>0.9247</b></p>
            <p>High accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #4CAF50;">👥 Top Nodes</h3>
            <p><b>30 Nodes</b></p>
            <p>Available for testing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #FF9800;">🔍 Features</h3>
            <p><b>20+ Features</b></p>
            <p>Graph-based metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Project Details Section
    st.markdown('<div class="sub-header">📊 Project Details</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Project Objective</h3>
        <p>Develop a machine learning model to predict missing links in a social network graph, 
        specifically for friend recommendations on Facebook.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔧 Technical Approach
        
        **1. Graph Feature Engineering:**
        - Jaccard Similarity
        - Common Neighbors
        - Preferential Attachment
        - Page Rank scores
        - SVD features
        
        **2. Machine Learning Models:**
        - Random Forest (Final Model)
        - XGBoost 
        - Logistic Regression
        
        **3. Evaluation Metrics:**
        - F1 Score
        - Precision & Recall
        - ROC-AUC
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Key Features Used
        
        **Graph-Based Features:**
        - `jaccard_followers`
        - `jaccard_followees` 
        - `cosine_followers`
        - `cosine_followees`
        - `num_followers_s/d`
        - `num_followees_s/d`
        
        **Advanced Features:**
        - `preferential_attachment`
        - `svd_dot_product`
        - `page_rank_source/dest`
        - `shortest_path`
        """)
    
    # Model Performance
    st.markdown("""
    ### 🏆 Model Performance
    """)
    
    performance_data = {
        'Metric': ['F1 Score', 'Precision', 'Recall', 'AUC'],
        'Train': [0.9251, 0.9245, 0.9257, 0.9856],
        'Test': [0.9247, 0.9241, 0.9253, 0.9854]
    }
    
    perf_df = pd.DataFrame(performance_data)
    st.table(perf_df)
    
    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Kaggle Competition Section
    st.markdown('<div class="sub-header">🏆 Kaggle Competition Information</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>📢 Competition: Facebook Friend Recommendation</h3>
        <p><b>Hosted by:</b> Facebook (Meta) - Social Computing Team</p>
        <p><b>Competition Type:</b> Link Prediction Challenge</p>
        <p><b>Dataset:</b> Facebook Social Network Graph</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Problem Statement
        
        **Objective:** Predict missing edges in a social network graph
        
        **Challenge:** Given a directed social graph, predict which users are likely to form new friendships.
        
        **Data:** 
        - Source and destination user pairs
        - Social graph features (followers, followees)
        - Binary labels (0 = no connection, 1 = connection)
        
        **Evaluation:** F1 Score on test predictions
        """)
    
    with col2:
        st.markdown("""
        ### 💼 Business Impact
        
        **Why This Matters:**
        - Increase user engagement
        - Improve platform stickiness
        - Enhance user experience
        - Drive ad revenue through better connections
        
        **Current Solutions:**
        - Mutual friends suggestions
        - Location-based recommendations
        - Interest similarity matching
        - Machine learning ranked suggestions
        """)
    
    st.markdown("""
    <div class="success-box">
        <h3>🏅 Our Solution Advantages</h3>
        <ul>
            <li><b>High Accuracy:</b> 92.47% F1 Score</li>
            <li><b>Graph-Based:</b> Leverages network topology</li>
            <li><b>Scalable:</b> Efficient Random Forest implementation</li>
            <li><b>Interpretable:</b> Clear feature importance</li>
            <li><b>Real-time:</b> Fast prediction capabilities</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Friend Recommendation Section
    st.markdown('<div class="sub-header">🤖 Friend Recommendation Engine</div>', unsafe_allow_html=True)
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h3 style="margin-bottom: 0rem; margin-top: 0rem;">👤 Enter Friend ID for Recommendations</h3>', unsafe_allow_html=True)
    
    # Use number input that updates session state
    user_input = st.number_input(
        "Friend ID:", 
        min_value=1, 
        max_value=2000000,
        value=st.session_state.selected_node,
        help="Enter a valid Friend ID to get recommendations",
        key="friend_id_input"
    )
    
    # Update session state when user types manually
    if user_input != st.session_state.selected_node:
        st.session_state.selected_node = user_input
    
    num_recommendations = st.slider(
        "Number of recommendations:", 
        min_value=1, 
        max_value=10, 
        value=5
    )
    
    recommend_button = st.button("🔍 Get Friend Recommendations", type="primary")
    
    
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>💡 Try These Top Nodes:</h4>
            <p>Use these verified Friend IDs for best results:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display top performing nodes with working buttons
        top_nodes = top_30_df.head(10)['node_id'].tolist()
        for i, node in enumerate(top_nodes, 1):
            if st.button(f"🎯 :blue[Try ID: {node}]", key=f"try_node_{node}_{i}"):
                st.session_state.selected_node = int(node)
                st.rerun()
    
    # Update user_input if session state changed
    if st.session_state.selected_node != user_input:
        user_input = st.session_state.selected_node
    
    # Recommendations section
    if recommend_button:
        with st.spinner("🔄 Generating recommendations..."):
            result = recommend_friends(user_input, df_test, model, num_recommendations)
            
            if result['status'] == 'Success':
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ Recommendations for Friend ID: {user_input}</h3>
                    <p><b>Total Candidates Found:</b> {result['total_candidates']}</p>
                    <p><b>Actual Friends in Test Set:</b> {result['actual_friends']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display recommendations
                st.markdown("### 👥 Recommended Friends")
                
                cols = st.columns(min(num_recommendations, 3))
                for i, rec in enumerate(result['recommendations']):
                    with cols[i % 3]:
                        display_friend_icon(
                            rec['destination_node'],
                            rec['friendship_probability'],
                            rec['confidence'],
                            rec['actual_connection']
                        )
                
                # Accuracy metrics
                correct_predictions = sum(1 for r in result['recommendations'] if r['actual_connection'] == 1)
                accuracy = (correct_predictions / len(result['recommendations']) * 100) if result['recommendations'] else 0
                
                st.markdown(f"""
                ### 📊 Recommendation Quality
                - **Accuracy:** {accuracy:.1f}% ({correct_predictions}/{len(result['recommendations'])})
                - **Average Confidence:** {np.mean([r['friendship_probability'] for r in result['recommendations']]):.3f}
                """)
                
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h3>⚠️ No Recommendations Available</h3>
                    <p>Friend ID {user_input} has no potential connections in our test dataset.</p>
                    <p>Please try one of the suggested Friend IDs from the right panel.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Display top 30 nodes for reference
    st.markdown("### 🎯 Top 30 Warm Start Nodes")
    st.markdown("These nodes have been verified to have potential friend connections:")
    
    # Create a nice display of all 30 nodes
    node_cols = st.columns(6)
    for i, (_, row) in enumerate(top_30_df.iterrows()):
        with node_cols[i % 6]:
            accuracy = row['friend_ratio'] * 100
            st.markdown(f"""
            <div class="node-badge">
                ID: {int(row['node_id'])}<br>
                Friends: {int(row['actual_friends'])}<br>
                Accuracy: {accuracy:.0f}%
            </div>
            """, unsafe_allow_html=True)
    
    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Analytics Dashboard Section
    st.markdown('<div class="sub-header">📈 Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Top 30 nodes analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Friend ratio distribution
        fig = px.histogram(
            top_30_df, 
            x='friend_ratio', 
            title='Friend Ratio Distribution',
            labels={'friend_ratio': 'Friend Ratio', 'count': 'Number of Nodes'},
            color_discrete_sequence=['#1877F2']
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Total candidates vs actual friends
        fig = px.scatter(
            top_30_df, 
            x='total_candidates', 
            y='actual_friends',
            title='Candidates vs Actual Friends',
            labels={'total_candidates': 'Total Candidates', 'actual_friends': 'Actual Friends'},
            color='friend_ratio',
            size='avg_probability',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, width='stretch')
    
    # Detailed table
    st.markdown("### 📋 Detailed Node Analysis")
    display_df = top_30_df.copy()
    display_df['friend_ratio'] = display_df['friend_ratio'].round(3)
    display_df['avg_probability'] = display_df['avg_probability'].round(3)
    
    st.dataframe(
        display_df,
        width='stretch',
        column_config={
            "node_id": "Node ID",
            "total_candidates": "Total Candidates",
            "actual_friends": "Actual Friends",
            "high_confidence": "High Confidence",
            "medium_confidence": "Medium Confidence",
            "friend_ratio": "Friend Ratio",
            "avg_probability": "Avg Probability"
        }
    )
    
    # Summary statistics
    st.markdown("### 📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_candidates = top_30_df['total_candidates'].sum()
        st.metric("Total Candidates", f"{total_candidates:,}")
    
    with col2:
        total_friends = top_30_df['actual_friends'].sum()
        st.metric("Total Actual Friends", f"{total_friends:,}")
    
    with col3:
        avg_ratio = top_30_df['friend_ratio'].mean()
        st.metric("Average Friend Ratio", f"{avg_ratio:.3f}")
    
    with col4:
        perfect_nodes = len(top_30_df[top_30_df['friend_ratio'] == 1.0])
        st.metric("Perfect Ratio Nodes", f"{perfect_nodes}/30")

if __name__ == "__main__":

    main()


