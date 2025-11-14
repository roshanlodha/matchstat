###############################################
# Imports
###############################################

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###############################################
# Data structures
###############################################


@dataclass
class MatchSimulationResult:
    effective_ranks: np.ndarray
    difficulty_scores: np.ndarray
    p_program: np.ndarray
    p_match_at_rank: np.ndarray
    cumulative_match_prob: np.ndarray
    p_unmatched: float
    total_prob: float
    p_match_anywhere: float
    e_rank_with_penalty: float
    e_rank_given_match: float
    e_fall_distance_given_match: float
    df_summary: pd.DataFrame


###############################################
# Core simulation
###############################################


def simulate_match(
    step2_score: float = 268,
    cclcm_penalty: float = 0,
    num_publications: int = 16,
    first_authors: int = 4,
    h_index: int = 4,
    num_programs: int = 20,
    unmatched_penalty_rank: int = 25,
    alpha: float = -0.25,
    beta_step2: float = 1.5 * math.log(2) / 10.0,
    gamma_cclcm_penalty: float = 0.40,
    gamma_research: float = 0.40,
    gamma_home: float = 0.50,
    w_pubs: float = 0.50,
    w_first_author: float = 0.35,
    w_h_index: float = 0.15,
    base_difficulty: float = 0.0,
    difficulty_amplitude: float = 2.0,
    decay_rate: float = 0.25,
    difficulty_noise_std: float = 0.1,
    decay_power: float = 1.5,
    random_seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> MatchSimulationResult:
    if rng is not None and random_seed is not None:
        raise ValueError("Provide either rng or random_seed, not both.")
    if rng is None:
        rng = np.random.default_rng(random_seed)

    z_step2 = (step2_score - 200.0) / 10.0
    logit_step2 = beta_step2 * z_step2
    logit_cclcm = -gamma_cclcm_penalty * cclcm_penalty

    research_score = (
        w_pubs * np.log1p(num_publications)
        + w_first_author * np.log1p(first_authors)
        + w_h_index * np.log1p(h_index)
    )
    logit_research = gamma_research * research_score
    logit_home = gamma_home * 1.0

    S = logit_step2 + logit_cclcm + logit_research + logit_home

    ranks = np.arange(1, num_programs + 1)
    difficulty_decay_part = difficulty_amplitude * np.exp(
        -decay_rate * (ranks - 1) ** decay_power
    )
    difficulty_noise = rng.normal(loc=0.0, scale=difficulty_noise_std, size=num_programs)
    difficulty_scores = base_difficulty + difficulty_decay_part + difficulty_noise

    logits = alpha + S - difficulty_scores
    p_program = 1.0 / (1.0 + np.exp(-logits))
    eps = 1e-9
    p_program = np.clip(p_program, eps, 1.0 - eps)

    one_minus_p = 1.0 - p_program

    cum_fail = np.empty(num_programs + 1)
    cum_fail[0] = 1.0
    for k in range(1, num_programs + 1):
        cum_fail[k] = cum_fail[k - 1] * one_minus_p[k - 1]

    p_match_at_rank = np.empty(num_programs)
    for k in range(1, num_programs + 1):
        p_match_at_rank[k - 1] = cum_fail[k - 1] * p_program[k - 1]

    p_unmatched = cum_fail[num_programs]
    total_prob = p_match_at_rank.sum() + p_unmatched

    effective_ranks = ranks.astype(float)
    e_rank_with_penalty = (
        (effective_ranks * p_match_at_rank).sum() + unmatched_penalty_rank * p_unmatched
    )
    p_match_anywhere = 1.0 - p_unmatched
    e_rank_given_match = ((effective_ranks * p_match_at_rank).sum()) / p_match_anywhere
    e_fall_distance_given_match = e_rank_given_match - 1.0

    cumulative_match_prob = np.cumsum(p_match_at_rank)

    df = pd.DataFrame(
        {
            "Rank": effective_ranks,
            "DifficultyScore": difficulty_scores,
            "ProgramMatchProb": p_program,
            "ProbMatchAtThisRank": p_match_at_rank,
            "CumulativeProbMatchedByThisRank": cumulative_match_prob,
        }
    )

    unmatched_row = pd.DataFrame(
        {
            "Rank": ["Unmatched"],
            "DifficultyScore": [np.nan],
            "ProgramMatchProb": [np.nan],
            "ProbMatchAtThisRank": [p_unmatched],
            "CumulativeProbMatchedByThisRank": [np.nan],
        }
    )

    df_summary = pd.concat([df, unmatched_row], ignore_index=True)

    return MatchSimulationResult(
        effective_ranks=effective_ranks,
        difficulty_scores=difficulty_scores,
        p_program=p_program,
        p_match_at_rank=p_match_at_rank,
        cumulative_match_prob=cumulative_match_prob,
        p_unmatched=p_unmatched,
        total_prob=total_prob,
        p_match_anywhere=p_match_anywhere,
        e_rank_with_penalty=e_rank_with_penalty,
        e_rank_given_match=e_rank_given_match,
        e_fall_distance_given_match=e_fall_distance_given_match,
        df_summary=df_summary,
    )


###############################################
# Plotting helpers
###############################################


def _plot_difficulty(result: MatchSimulationResult, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.effective_ranks, result.difficulty_scores, marker="o")
    ax.set_xlabel("Rank position on list")
    ax.set_ylabel("Program difficulty score (higher = harder)")
    ax.set_title("Program difficulty distribution along rank list")
    ax.grid(True)
    fig.tight_layout()
    output_path = output_dir / "program_difficulty.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _plot_per_program_prob(result: MatchSimulationResult, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.effective_ranks, result.p_program, marker="o")
    ax.set_xlabel("Rank position on list")
    ax.set_ylabel("Per-program match probability p_i")
    ax.set_title("Per-program match probability vs rank")
    ax.grid(True)
    fig.tight_layout()
    output_path = output_dir / "per_program_match_prob.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _plot_match_distribution(result: MatchSimulationResult, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(result.effective_ranks, result.p_match_at_rank)
    ax.set_xlabel("Rank position on list")
    ax.set_ylabel("Probability of matching at this exact rank")
    ax.set_title("Distribution of match rank")
    ax.axvline(
        x=result.e_rank_given_match,
        color="k",
        linestyle="--",
        label=f"Expected rank (given match) = {result.e_rank_given_match:.2f}",
    )
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / "match_rank_distribution.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _plot_cumulative_prob(result: MatchSimulationResult, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.effective_ranks, result.cumulative_match_prob, marker="o")
    ax.set_xlabel("Rank position on list")
    ax.set_ylabel("Cumulative probability of being matched by this rank")
    ax.set_title("Cumulative match probability vs rank")
    ax.set_ylim(0, 1.0)
    ax.grid(True)
    fig.tight_layout()
    output_path = output_dir / "cumulative_match_probability.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_plots(result: MatchSimulationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_difficulty(result, output_dir)
    _plot_per_program_prob(result, output_dir)
    _plot_match_distribution(result, output_dir)
    _plot_cumulative_prob(result, output_dir)


###############################################
# Persistence helpers
###############################################


def save_summary_table(result: MatchSimulationResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary_table.csv"
    result.df_summary.to_csv(output_path, index=False)
    return output_path


def save_summary_stats(result: MatchSimulationResult, output_dir: Path) -> Path:
    stats = {
        "total_probability": float(result.total_prob),
        "probability_match_anywhere": float(result.p_match_anywhere),
        "probability_unmatched": float(result.p_unmatched),
        "expected_rank_with_penalty": float(result.e_rank_with_penalty),
        "expected_rank_given_match": float(result.e_rank_given_match),
        "expected_fall_distance_given_match": float(result.e_fall_distance_given_match),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary_stats.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return output_path


###############################################
# CLI
###############################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate match outcomes.")
    parser.add_argument("--step2-score", type=float, default=268)
    parser.add_argument("--cclcm-penalty", type=float, default=0.0)
    parser.add_argument("--num-publications", type=int, default=16)
    parser.add_argument("--first-authors", type=int, default=4)
    parser.add_argument("--h-index", type=int, default=4)
    parser.add_argument("--student-name", type=str, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Base directory for simulation artifacts.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    student_dir = args.output_root / args.student_name
    student_dir.mkdir(parents=True, exist_ok=True)

    result = simulate_match(
        step2_score=args.step2_score,
        cclcm_penalty=args.cclcm_penalty,
        num_publications=args.num_publications,
        first_authors=args.first_authors,
        h_index=args.h_index,
        random_seed=args.random_seed,
    )

    save_plots(result, student_dir)
    table_path = save_summary_table(result, student_dir)
    stats_path = save_summary_stats(result, student_dir)

    print(f"Artifacts saved under: {student_dir}")
    print(f"Summary table -> {table_path}")
    print(f"Summary stats -> {stats_path}")


if __name__ == "__main__":
    main()
