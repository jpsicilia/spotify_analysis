"""Reusable plotting helpers (matplotlib / seaborn / plotly)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns


def barh_ranked(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    palette: str = "crest",
    figsize: tuple[int, int] = (12, 6),
) -> plt.Axes:
    """Horizontal bar chart of a ranked table, with value labels."""
    plt.figure(figsize=figsize)
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=data, x=x, y=y, hue=y, palette=palette, legend=False)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(y.capitalize(), fontsize=12)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=10)
    plt.tight_layout()
    return ax


def build_cooccurrence_figure(pair_counts, min_weight: int = 1) -> go.Figure:
    """Build the interactive Plotly co-occurrence network.

    Args:
        pair_counts: Counter of ``(g1, g2) -> weight`` from
            :func:`spotify_analysis.analysis.genre_cooccurrence`.
        min_weight: Drop edges below this weight to reduce clutter.

    Returns:
        A Plotly Figure ready to ``.show()``.
    """
    graph = nx.Graph()
    for (g1, g2), weight in pair_counts.items():
        if weight >= min_weight:
            graph.add_edge(g1, g2, weight=weight)

    pos = nx.spring_layout(graph, k=0.4, iterations=50, seed=42)

    edge_x, edge_y = [], []
    for a, b in graph.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, node_size, node_text = [], [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = graph.degree(node)
        node_size.append(10 + degree * 2)
        neighbors = sorted(
            ((n, graph[node][n]["weight"]) for n in graph.neighbors(node)),
            key=lambda t: t[1],
            reverse=True,
        )[:3]
        conns = "<br>".join(f"{n}: {w} songs" for n, w in neighbors)
        node_text.append(
            f"<b>{node}</b><br>Connected to {degree} genres<br>"
            f"Top connections:<br>{conns}"
        )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=1, color="#888"),
        hoverinfo="none", mode="lines",
    )
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=[n.capitalize() for n in graph.nodes()],
        textposition="top center", hovertext=node_text, hoverinfo="text",
        marker=dict(
            showscale=True, colorscale="YlGnBu", size=node_size,
            color=[graph.degree(n) for n in graph.nodes()],
            colorbar=dict(thickness=15, title="Connections"),
            line_width=2,
        ),
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="<b>Genre co-occurrence network</b>",
        showlegend=False, hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=800,
    )
    return fig
