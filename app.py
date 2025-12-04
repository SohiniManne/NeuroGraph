import streamlit as st
import torch
import pandas as pd
import time

# Modular Imports
from src.model import train_model
from src.data_gen import generate_social_graph
from src.visualizer import plot_network

# --- CONFIG ---
st.set_page_config(page_title="NeuroGraph | Inductive AI", layout="wide", page_icon="🕸️")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1 { color: #4F8BF9; }
    .stButton>button { border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'data' not in st.session_state:
    st.session_state['data'], st.session_state['graph'], st.session_state['names'] = generate_social_graph()

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ NeuroGraph Ops")
    
    st.markdown("### Hyperparameters")
    n_nodes = st.slider("Graph Nodes", 50, 500, 150)
    emb_size = st.slider("Embedding Dim", 16, 128, 64)
    epochs = st.slider("Training Epochs", 20, 200, 60)
    
    if st.button("Regenerate Graph"):
        st.session_state['data'], st.session_state['graph'], st.session_state['names'] = generate_social_graph(n_nodes)
        st.rerun()  # <--- UPDATED: Changed from experimental_rerun() to rerun()

# --- MAIN PAGE ---
st.title("🕸️ NeuroGraph: Inductive Social Inference")
st.markdown("##### Real-time link prediction using GraphSAGE on PyTorch Geometric")

tab1, tab2, tab3 = st.tabs(["📊 Network Topology", "🧠 Model Training", "🔮 Inference Engine"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        # We catch errors here to prevent the UI from crashing if plot fails
        try:
            st.plotly_chart(plot_network(st.session_state['graph']), use_container_width=True)
        except Exception as e:
            st.error(f"Visualization Error: {e}")
            
    with col2:
        st.write("### Graph Stats")
        st.metric("Users", st.session_state['data'].num_nodes)
        st.metric("Connections", st.session_state['data'].num_edges // 2)
        st.metric("Sparsity", f"{st.session_state['data'].num_edges / (n_nodes**2):.4f}")

with tab2:
    st.write("### GNN Training Pipeline")
    if st.button("Start Training Job", type="primary"):
        progress_bar = st.progress(0)
        status = st.empty()
        
        def update_ui(epoch, total_epochs, loss):
            progress_bar.progress(epoch / total_epochs)
            status.markdown(f"**Epoch {epoch}** | Loss: `{loss:.5f}`")
            
        model, losses = train_model(st.session_state['data'], emb_size, epochs, update_ui)
        
        st.session_state['model'] = model
        st.success(f"Model Converged! Final Loss: {losses[-1]:.4f}")
        
        # Analytics
        loss_df = pd.DataFrame(losses, columns=["Loss"])
        st.area_chart(loss_df)

with tab3:
    st.write("### Real-time Recommendation Interface")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Model not trained. Please go to the 'Model Training' tab.")
    else:
        target_user = st.selectbox("Select Target User", st.session_state['names'])
        user_idx = int(target_user.split("_")[1])
        
        if st.button("Generate Recommendations"):
            model = st.session_state['model']
            data = st.session_state['data']
            model.eval()
            
            with torch.no_grad():
                z = model.encode(data.x, data.edge_index)
                user_emb = z[user_idx]
                scores = torch.matmul(z, user_emb)
                
                G = st.session_state['graph']
                existing = set(G.neighbors(user_idx))
                existing.add(user_idx)
                
                predictions = []
                for i, score in enumerate(scores):
                    if i not in existing:
                        predictions.append((f"User_{i}", score.item()))
                
                predictions.sort(key=lambda x: x[1], reverse=True)
                top_5 = predictions[:5]
                
                st.subheader(f"Top 5 Suggested Friends for {target_user}")
                cols = st.columns(5)
                for idx, (name, score) in enumerate(top_5):
                    with cols[idx]:
                        confidence = min(max((score + 5) / 10, 0.0), 1.0)
                        st.metric(label=name, value=f"{score:.2f}")
                        st.progress(confidence)