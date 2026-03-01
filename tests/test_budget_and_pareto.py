import numpy as np

from src.budget import DynamicPruningBudget
from src.pareto import ParetoPoint, compute_pareto_frontier
from src.theory import analyze_lottery_ticket_behavior


def test_dynamic_budget_increases_with_complexity() -> None:
    controller = DynamicPruningBudget()
    easy = "What is 2+2?"
    hard = "Solve: ((123*45)-67)/89 and explain each algebraic transformation in detail."
    assert controller.compute_keep_ratio(hard) > controller.compute_keep_ratio(easy)


def test_pareto_frontier_filters_dominated_points() -> None:
    points = [
        ParetoPoint("a", accuracy=0.80, latency_ms=100.0, sparsity=0.4),
        ParetoPoint("b", accuracy=0.81, latency_ms=90.0, sparsity=0.3),
        ParetoPoint("c", accuracy=0.79, latency_ms=95.0, sparsity=0.5),
    ]
    result = compute_pareto_frontier(points)
    labels = [p.label for p in result.frontier]
    assert "b" in labels
    assert "a" not in labels


def test_lottery_ticket_analysis_outputs_valid_scores() -> None:
    masks = np.array(
        [
            [[0.9, 0.2], [0.8, 0.1]],
            [[0.85, 0.2], [0.75, 0.15]],
            [[0.9, 0.1], [0.8, 0.2]],
        ]
    )
    analysis = analyze_lottery_ticket_behavior(masks)
    assert 0.0 <= analysis.winning_ticket_score <= 1.0
    assert analysis.mean_overlap > 0.0
