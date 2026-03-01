"""Interactive dashboard for pruning experiments (Streamlit)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    st.set_page_config(page_title="Microglia Pruning Dashboard", layout="wide")
    st.title("Microglia Pruning Research Dashboard")

    st.sidebar.header("Inputs")
    benchmark_path = Path(st.sidebar.text_input("Benchmark JSON", "results/benchmark_suite_results.json"))
    pareto_path = Path(st.sidebar.text_input("Pareto JSON", "results/pareto_results.json"))

    if not benchmark_path.exists() or not pareto_path.exists():
        st.warning("Please run benchmark_suite.py and pareto_explorer.py first.")
        return

    benchmark = _load_json(benchmark_path)
    pareto = _load_json(pareto_path)

    runs = pd.DataFrame(benchmark.get("runs", []))
    points = pd.DataFrame(pareto.get("points", []))
    frontier = pd.DataFrame(pareto.get("frontier", []))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accuracy by Dataset and Mode")
        fig_acc = px.bar(runs, x="dataset", y="accuracy", color="mode", barmode="group")
        st.plotly_chart(fig_acc, use_container_width=True)

    with col2:
        st.subheader("Pareto Frontier")
        fig_pareto = px.scatter(
            points,
            x="latency_ms",
            y="accuracy",
            color="sparsity",
            hover_name="label",
            title="Accuracy-Latency Trade-off",
        )
        if not frontier.empty:
            fig_pareto.add_scatter(
                x=frontier["latency_ms"],
                y=frontier["accuracy"],
                mode="lines+markers",
                name="frontier",
            )
        st.plotly_chart(fig_pareto, use_container_width=True)

    st.subheader("Raw Tables")
    st.dataframe(runs)
    st.dataframe(points)


if __name__ == "__main__":
    main()
