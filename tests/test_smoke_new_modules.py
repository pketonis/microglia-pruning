import numpy as np

from src.budget import DynamicPruningBudget
from src.pareto import ParetoPoint, compute_pareto_frontier
from src.theory import analyze_lottery_ticket_behavior


def test_smoke_budget_module():
    b = DynamicPruningBudget()
    r = b.compute_keep_ratio("What is 2 + 2?")
    assert 0.0 < r <= 1.0


def test_smoke_pareto_module():
    result = compute_pareto_frontier([
        ParetoPoint("a", 0.7, 120.0, 0.5),
        ParetoPoint("b", 0.8, 100.0, 0.4),
    ])
    assert len(result.frontier) >= 1


def test_smoke_theory_module():
    masks = np.ones((3, 2, 4), dtype=float)
    analysis = analyze_lottery_ticket_behavior(masks)
    assert analysis.winning_ticket_score == 1.0
