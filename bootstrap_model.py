import argparse
import json
import math
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from tqdm import trange
except ImportError:  # pragma: no cover - tqdm is optional
    def trange(n, *_, **__):
        return range(n)

from model import MatchSimulationResult, simulate_match

Z_VALUE = 1.96  # 95% confidence interval


def summarize_array(values: np.ndarray) -> Dict[str, np.ndarray]:
    mean = values.mean(axis=0)
    if values.shape[0] > 1:
        std = values.std(axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)
    if values.shape[0] > 0:
        se = std / math.sqrt(values.shape[0])
    else:
        se = np.zeros_like(mean)
    ci_half = Z_VALUE * se
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - ci_half,
        "ci_high": mean + ci_half,
    }


def summarize_scalar(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"mean": math.nan, "std": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    mean = float(values.mean())
    if values.size > 1:
        std = float(values.std(ddof=1))
    else:
        std = 0.0
    se = std / math.sqrt(values.size) if values.size > 0 else 0.0
    ci_half = Z_VALUE * se
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - ci_half,
        "ci_high": mean + ci_half,
    }


def build_bootstrap_table(
    ranks: np.ndarray,
    difficulty_stats: Dict[str, np.ndarray],
    per_program_stats: Dict[str, np.ndarray],
    match_rank_stats: Dict[str, np.ndarray],
    cumulative_stats: Dict[str, np.ndarray],
    unmatched_stats: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    for idx, rank in enumerate(ranks):
        rows.append(
            {
                "Rank": int(rank),
                "DifficultyScoreMean": difficulty_stats["mean"][idx],
                "DifficultyScoreStd": difficulty_stats["std"][idx],
                "DifficultyScoreCILow": difficulty_stats["ci_low"][idx],
                "DifficultyScoreCIHigh": difficulty_stats["ci_high"][idx],
                "ProgramMatchProbMean": per_program_stats["mean"][idx],
                "ProgramMatchProbStd": per_program_stats["std"][idx],
                "ProgramMatchProbCILow": per_program_stats["ci_low"][idx],
                "ProgramMatchProbCIHigh": per_program_stats["ci_high"][idx],
                "ProbMatchAtThisRankMean": match_rank_stats["mean"][idx],
                "ProbMatchAtThisRankStd": match_rank_stats["std"][idx],
                "ProbMatchAtThisRankCILow": match_rank_stats["ci_low"][idx],
                "ProbMatchAtThisRankCIHigh": match_rank_stats["ci_high"][idx],
                "CumulativeProbMatchedByThisRankMean": cumulative_stats["mean"][idx],
                "CumulativeProbMatchedByThisRankStd": cumulative_stats["std"][idx],
                "CumulativeProbMatchedByThisRankCILow": cumulative_stats["ci_low"][idx],
                "CumulativeProbMatchedByThisRankCIHigh": cumulative_stats["ci_high"][idx],
            }
        )

    rows.append(
        {
            "Rank": "Unmatched",
            "DifficultyScoreMean": math.nan,
            "DifficultyScoreStd": math.nan,
            "DifficultyScoreCILow": math.nan,
            "DifficultyScoreCIHigh": math.nan,
            "ProgramMatchProbMean": math.nan,
            "ProgramMatchProbStd": math.nan,
            "ProgramMatchProbCILow": math.nan,
            "ProgramMatchProbCIHigh": math.nan,
            "ProbMatchAtThisRankMean": unmatched_stats["mean"],
            "ProbMatchAtThisRankStd": unmatched_stats["std"],
            "ProbMatchAtThisRankCILow": unmatched_stats["ci_low"],
            "ProbMatchAtThisRankCIHigh": unmatched_stats["ci_high"],
            "CumulativeProbMatchedByThisRankMean": math.nan,
            "CumulativeProbMatchedByThisRankStd": math.nan,
            "CumulativeProbMatchedByThisRankCILow": math.nan,
            "CumulativeProbMatchedByThisRankCIHigh": math.nan,
        }
    )

    return pd.DataFrame(rows)


def _error_bars(mean: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray) -> np.ndarray:
    lower = mean - ci_low
    upper = ci_high - mean
    return np.vstack([lower, upper])


def save_bootstrap_plots(
    ranks: np.ndarray,
    difficulty_stats: Dict[str, np.ndarray],
    per_program_stats: Dict[str, np.ndarray],
    match_rank_stats: Dict[str, np.ndarray],
    cumulative_stats: Dict[str, np.ndarray],
    e_rank_given_match_mean: float,
    e_rank_given_match_ci: Dict[str, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(
        ranks,
        difficulty_stats["mean"],
        yerr=_error_bars(
            difficulty_stats["mean"], difficulty_stats["ci_low"], difficulty_stats["ci_high"]
        ),
        fmt="-o",
        capsize=4,
    )
    ax.set_xlabel("Rank position on list")
    ax.set_ylabel("Program difficulty score (higher = harder)")
    ax.set_title("Bootstrapped program difficulty")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "bootstrap_program_difficulty.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(
        ranks,
        per_program_stats["mean"],
        yerr=_error_bars(
            per_program_stats["mean"],
            per_program_stats["ci_low"],
            per_program_stats["ci_high"],
        ),
        fmt="-o",
        capsize=4,
    )
    ax.set_xlabel("Rank position on list")
    ax.set_ylabel("Per-program match probability p_i")
    ax.set_ylim(0, 1)
    ax.set_title("Bootstrapped per-program match probability")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "bootstrap_per_program_match_prob.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ranks, match_rank_stats["mean"], color="#5b8dc2", alpha=0.8)
    ax.errorbar(
        ranks,
        match_rank_stats["mean"],
        yerr=_error_bars(
            match_rank_stats["mean"],
            match_rank_stats["ci_low"],
            match_rank_stats["ci_high"],
        ),
        fmt="none",
        ecolor="k",
        capsize=4,
    )
    ax.set_xlabel("Rank position on list")
    ax.set_ylabel("Probability of matching at this exact rank")
    ax.set_title("Bootstrapped match rank distribution")
    ax.axvline(
        x=e_rank_given_match_mean,
        color="k",
        linestyle="--",
        label=f"Mean expected rank (given match) = {e_rank_given_match_mean:.2f}",
    )
    ax.plot(
        [],
        [],
        " ",
        label=(
            f"Expected rank 95% CI: "
            f"[{e_rank_given_match_ci['ci_low']:.2f}, {e_rank_given_match_ci['ci_high']:.2f}]"
        ),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bootstrap_match_rank_distribution.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(
        ranks,
        cumulative_stats["mean"],
        yerr=_error_bars(
            cumulative_stats["mean"],
            cumulative_stats["ci_low"],
            cumulative_stats["ci_high"],
        ),
        fmt="-o",
        capsize=4,
    )
    ax.set_xlabel("Rank position on list")
    ax.set_ylabel("Cumulative probability of being matched by this rank")
    ax.set_title("Bootstrapped cumulative match probability")
    ax.set_ylim(0, 1)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "bootstrap_cumulative_match_probability.png", dpi=300)
    plt.close(fig)


def save_bootstrap_stats(stats: Dict[str, Dict[str, float]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "bootstrapped_summary_stats.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap match model outputs.")
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
        "--num-runs",
        type=int,
        default=250,
        help="Number of bootstrap iterations.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible bootstraps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_runs <= 0:
        raise ValueError("num-runs must be positive.")

    student_dir = args.output_root / args.student_name
    bootstrap_dir = student_dir / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_seed)
    results = []
    for _ in trange(args.num_runs, desc="Bootstrapping"):
        res = simulate_match(
            step2_score=args.step2_score,
            cclcm_penalty=args.cclcm_penalty,
            num_publications=args.num_publications,
            first_authors=args.first_authors,
            h_index=args.h_index,
            rng=rng,
        )
        results.append(res)

    difficulty_stack = np.stack([res.difficulty_scores for res in results])
    per_program_stack = np.stack([res.p_program for res in results])
    match_rank_stack = np.stack([res.p_match_at_rank for res in results])
    cumulative_stack = np.stack([res.cumulative_match_prob for res in results])

    difficulty_stats = summarize_array(difficulty_stack)
    per_program_stats = summarize_array(per_program_stack)
    match_rank_stats = summarize_array(match_rank_stack)
    cumulative_stats = summarize_array(cumulative_stack)

    unmatched_values = np.array([res.p_unmatched for res in results])
    match_anywhere_values = np.array([res.p_match_anywhere for res in results])
    total_prob_values = np.array([res.total_prob for res in results])
    e_rank_with_penalty_vals = np.array([res.e_rank_with_penalty for res in results])
    e_rank_given_match_vals = np.array([res.e_rank_given_match for res in results])
    e_fall_distance_vals = np.array([res.e_fall_distance_given_match for res in results])

    unmatched_stats = summarize_scalar(unmatched_values)
    stats = {
        "total_probability": summarize_scalar(total_prob_values),
        "probability_match_anywhere": summarize_scalar(match_anywhere_values),
        "probability_unmatched": unmatched_stats,
        "expected_rank_with_penalty": summarize_scalar(e_rank_with_penalty_vals),
        "expected_rank_given_match": summarize_scalar(e_rank_given_match_vals),
        "expected_fall_distance_given_match": summarize_scalar(e_fall_distance_vals),
    }

    base_ranks = results[0].effective_ranks
    bootstrap_table = build_bootstrap_table(
        base_ranks,
        difficulty_stats,
        per_program_stats,
        match_rank_stats,
        cumulative_stats,
        unmatched_stats,
    )
    table_path = bootstrap_dir / "bootstrapped_summary_table.csv"
    bootstrap_table.to_csv(table_path, index=False)

    save_bootstrap_plots(
        base_ranks,
        difficulty_stats,
        per_program_stats,
        match_rank_stats,
        cumulative_stats,
        stats["expected_rank_given_match"]["mean"],
        stats["expected_rank_given_match"],
        bootstrap_dir,
    )

    stats_path = save_bootstrap_stats(stats, bootstrap_dir)

    print(f"Bootstrapped artifacts saved under: {bootstrap_dir}")
    print(f"Bootstrapped summary table -> {table_path}")
    print(f"Bootstrapped summary stats -> {stats_path}")


if __name__ == "__main__":
    main()
