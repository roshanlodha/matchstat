import numpy as np
import math

def compute_topk_prob(step2_score=268,
                      cclcm_penalty=0,
                      num_publications=16,
                      first_authors=4,
                      h_index=4,
                      num_programs=20,
                      alpha=-0.5,
                      beta_step2_mul=1.0,
                      beta_step2_base=(math.log(2) / 10.0),
                      gamma_research=0.12,
                      gamma_cclcm_penalty=0.40,
                      gamma_home=0.50,
                      w_pubs=0.50,
                      w_first_author=0.35,
                      w_h_index=0.15,
                      base_difficulty=0.0,
                      difficulty_amplitude=3.0,
                      decay_rate=0.25,
                      decay_power=1.5,
                      difficulty_noise_std=0.1,
                      random_seed=42,
                      k=3):
    beta_step2 = beta_step2_base * beta_step2_mul

    z_step2 = (step2_score - 200.0) / 10.0
    logit_step2 = beta_step2 * z_step2
    logit_cclcm = -gamma_cclcm_penalty * cclcm_penalty
    research_score = (
        w_pubs * np.log1p(num_publications) +
        w_first_author * np.log1p(first_authors) +
        w_h_index * np.log1p(h_index)
    )
    logit_research = gamma_research * research_score
    logit_home = gamma_home * 1.0
    S = logit_step2 + logit_cclcm + logit_research + logit_home

    np.random.seed(random_seed)
    ranks = np.arange(1, num_programs + 1)
    difficulty_decay_part = difficulty_amplitude * np.exp(-decay_rate * (ranks - 1) ** decay_power)
    difficulty_noise = np.random.normal(loc=0.0, scale=difficulty_noise_std, size=num_programs)
    difficulty_scores = base_difficulty + difficulty_decay_part + difficulty_noise

    logits = alpha + S - difficulty_scores
    p_program = 1.0 / (1.0 + np.exp(-logits))
    p_program = np.clip(p_program, 1e-9, 1 - 1e-9)

    one_minus_p = 1.0 - p_program
    cum_fail = np.empty(num_programs + 1)
    cum_fail[0] = 1.0
    for idx in range(1, num_programs + 1):
        cum_fail[idx] = cum_fail[idx - 1] * one_minus_p[idx - 1]
    p_match_at_rank = np.empty(num_programs)
    for idx in range(1, num_programs + 1):
        p_match_at_rank[idx - 1] = cum_fail[idx - 1] * p_program[idx - 1]
    cumulative = np.cumsum(p_match_at_rank)
    topk = cumulative[k-1]
    return topk, p_program, difficulty_scores, cumulative

if __name__ == '__main__':
    # small grid search
    gamma_research_vals = [0.12, 0.2, 0.3, 0.4, 0.5]
    beta_mul_vals = [1.0, 1.5, 2.0]
    difficulty_amplitude_vals = [3.0, 2.5, 2.0, 1.5]
    alpha_vals = [-0.5, -0.25, 0.0]

    results = []
    for gr in gamma_research_vals:
        for bm in beta_mul_vals:
            for da in difficulty_amplitude_vals:
                for a in alpha_vals:
                    top3, p_program, diff_scores, cum = compute_topk_prob(alpha=a,
                                                                          beta_step2_mul=bm,
                                                                          gamma_research=gr,
                                                                          difficulty_amplitude=da)
                    results.append((top3, gr, bm, da, a))

    # sort by top3 desc
    results.sort(key=lambda x: x[0], reverse=True)
    print('Top results (top-3 cumulative prob, gamma_research, beta_mult, diff_ampl, alpha):')
    for r in results[:15]:
        print(r)

    # print best single candidate
    best = results[0]
    print('\nBest candidate -> top3 = {:.3f}, gamma_research={}, beta_mul={}, diff_amp={}, alpha={}'.format(*best))
