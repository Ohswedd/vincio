"""Vincio optimization engine."""

from .budget_learning import BudgetLearner, LearnedAllocations
from .cache_tuning import (
    CacheAdvice,
    CacheTuningReport,
    analyze_ast_layout,
    analyze_prompt_cacheability,
    cache_hit_economics,
)
from .context_search import ContextOptimizer, ContextSearchSpace
from .loop import DEFAULT_LOOP_METRICS, ImprovementLoop, LoopResult
from .pareto import (
    AGENTIC_OBJECTIVES,
    DEFAULT_OBJECTIVES,
    ObjectiveSpec,
    ParetoFrontier,
    ParetoPoint,
    ParetoResult,
    dominates,
    objective_vector,
    pareto_loop,
)
from .prompt_search import PromptOptimizer
from .retrieval_feedback import (
    ChunkingRecommendation,
    RelevanceRecord,
    RetrievalFeedback,
    RetrievalFeedbackResult,
    recommend_chunking,
    records_from_dataset,
    records_from_report,
)
from .routing import (
    EpsilonGreedyBandit,
    RoutingOptimizer,
    RoutingPolicy,
    UCB1Bandit,
    estimate_difficulty,
)
from .search import Candidate, FitnessWeights, OptimizationResult, evolution_loop, fitness
from .strategies import (
    AnnealingSearch,
    HillClimbSearch,
    RandomSearch,
    SearchStrategy,
    build_strategy,
    guided_search,
)

__all__ = [
    "BudgetLearner",
    "LearnedAllocations",
    "CacheAdvice",
    "CacheTuningReport",
    "analyze_ast_layout",
    "analyze_prompt_cacheability",
    "cache_hit_economics",
    "ContextOptimizer",
    "ContextSearchSpace",
    "DEFAULT_LOOP_METRICS",
    "ImprovementLoop",
    "LoopResult",
    "DEFAULT_OBJECTIVES",
    "AGENTIC_OBJECTIVES",
    "ObjectiveSpec",
    "ParetoFrontier",
    "ParetoPoint",
    "ParetoResult",
    "dominates",
    "objective_vector",
    "pareto_loop",
    "PromptOptimizer",
    "ChunkingRecommendation",
    "RelevanceRecord",
    "RetrievalFeedback",
    "RetrievalFeedbackResult",
    "recommend_chunking",
    "records_from_dataset",
    "records_from_report",
    "EpsilonGreedyBandit",
    "RoutingOptimizer",
    "RoutingPolicy",
    "UCB1Bandit",
    "estimate_difficulty",
    "Candidate",
    "FitnessWeights",
    "OptimizationResult",
    "evolution_loop",
    "fitness",
    "AnnealingSearch",
    "HillClimbSearch",
    "RandomSearch",
    "SearchStrategy",
    "build_strategy",
    "guided_search",
]
