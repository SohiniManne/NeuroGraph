import plotly.graph_objects as go
import networkx as nx

def plot_network(G_nx):
    """
    Creates a high-performance interactive Plotly graph.
    """
    pos = nx.spring_layout(G_nx, seed=42)
    
    # Edge Trace
    edge_x = []
    edge_y = []
    for edge in G_nx.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Node Trace
    node_x = []
    node_y = []
    node_text = []
    node_adj = []
    
    for node in G_nx.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        neighbors = len(list(G_nx.neighbors(node)))
        node_adj.append(neighbors)
        node_text.append(f"User ID: {node}<br>Friends: {neighbors}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=node_adj,
            size=12,
            colorbar=dict(
                thickness=15,
                title='Degree',
                xanchor='left'
            ),
            line_width=2))

    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    return fig