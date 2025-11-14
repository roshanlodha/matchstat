# Methods Overview

This document explains the modeling approach used in `model.py` and `bootstrap_model.py`. The goal is to approximate the distribution of match outcomes for a single applicant given an ordered rank list of residency programs.

## Applicant Strength

Applicant strength is summarized as a scalar score $S$ built from academic and research signals:

- **Step 2 CK:** normalized as $z_{\text{Step2}} = ( \text{Step2} - 200 ) / 10$.
- **CCLCM penalty:** a negative adjustment applied directly.
- **Research score:** weighted log-transformed counts of publications, first-author papers, and \( h \)-index.
- **Home program bonus:** a fixed positive term.

The net score is
$$
S = \beta_{\text{Step2}} z_{\text{Step2}}
    - \gamma_{\text{CCLCM}} \cdot \text{penalty}
    + \gamma_{\text{research}} \cdot \text{research\_score}
    + \gamma_{\text{home}}.
$$

## Program Difficulty Profile

Each rank position $r$ receives a baseline difficulty that decays with rank:
$$
D_r = \text{base} + A \cdot \exp\!\bigl(-\lambda (r-1)^{p}\bigr) + \varepsilon_r,
$$
where $A$ is the amplitude, $\lambda$ is the decay rate, $p$ controls curvature, and $\varepsilon_r \sim \mathcal{N}(0, \sigma^2)$ introduces heterogeneity between similarly ranked programs.

## Per-Program Match Probability

The probability of receiving an offer from rank $r$ is modeled with a logistic link:
$$
p_r = \frac{1}{1 + \exp(-( \alpha + S - D_r ))},
$$
clipped to $[10^{-9}, 1 - 10^{-9}]$ for numerical stability. Here $\alpha$ is a global intercept calibrated from tuning experiments.

## Rank Outcome Distribution

Assuming independence between programs, the probability of matching **exactly** at rank $k$ is
$$
\Pr(K = k) = \left( \prod_{j < k} (1 - p_j) \right) p_k,
$$
and the probability of going unmatched is
$$
\Pr(\text{unmatched}) = \prod_{j=1}^{N} (1 - p_j).
$$

These probabilities allow computation of:

- Expected rank with an unmatched penalty (assigning a synthetic rank to unmatched outcomes).
- Expected rank given a successful match.
- Expected “fall distance” $E[K - 1 \mid \text{match}]$ from the top choice.
- Cumulative probability of being matched by each rank position.

All tabular outputs are assembled into `summary_table.csv`, with an additional JSON file summarizing the scalar expectations described above.

## Visualization Outputs

`model.py` saves four plots per student under `outputs/<student_name>/`:

1. Difficulty profile across the rank list.
2. Per-program match probability.
3. Distribution of match rank, including the expected rank marker.
4. Cumulative match probability across ranks.

These figures are intended for rapid inspection of the simulated draw.

## Bootstrapping Methodology

`bootstrap_model.py` repeatedly calls the simulator to capture uncertainty introduced by the stochastic difficulty noise. For each bootstrap iteration:

1. A shared random number generator (`numpy.random.Generator`) produces a fresh noise sample for the entire rank list.
2. The resulting `MatchSimulationResult` is stored without writing intermediate artifacts.

After $n$ iterations (default $n = 250$):

- Arrays of difficulty scores, per-program probabilities, match-at-rank probabilities, and cumulative probabilities are stacked and summarized with means, standard deviations, and normal-approximation 95% confidence intervals ($\pm 1.96 \times \text{SE}$).
- Scalar statistics (total probability, unmatched probability, expected ranks, fall distance) receive the same treatment.
- The aggregated table (`bootstrapped_summary_table.csv`) mirrors the single-run table but includes mean/std/CI columns for every metric, plus the unmatched row.
- Four bootstrapped plots are created with error bars, and the match-rank distribution plot includes a dashed line for the mean expected rank along with its confidence interval in the legend.
- All bootstrap artifacts are saved under `outputs/<student_name>/bootstrap/`.

This process yields empirical variability estimates that can be used to place error bars on plots and to construct confidence intervals around the summary statistics, enabling clearer communication of uncertainty in the simulated applicant’s prospects.
