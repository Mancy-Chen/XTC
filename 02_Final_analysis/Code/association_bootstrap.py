"""Exploratory stratified participant bootstrap after within-group FDR screening."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def selected_outcomes(correlations, alpha=0.05):
    eligible = correlations.loc[
        correlations['sample'].isin(['XTC-naive', 'XTC users'])
        & correlations['adjustment'].eq('adjusted')
        & correlations['FDR_q'].lt(alpha), 'outcome'
    ]
    return sorted(eligible.unique())


def _rho(x, y):
    x = rankdata(x); y = rankdata(y)
    x -= x.mean(); y -= y.mean()
    den = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / den) if den > 0 else np.nan


def bootstrap_selected_correlations(data, correlations, roi_columns, output_dir,
                                    n_boot=10000, seed=20260901, alpha=0.05):
    output_dir = Path(output_dir)
    selected = selected_outcomes(correlations, alpha)
    rows = []; iterations = []
    for outcome in selected:
        if outcome == 'BrainSegVol':
            raw = 'raw_BrainSegVol'; predictors = ['brainseg_pre_cm3']
        else:
            base = roi_columns[outcome]
            raw = 'raw_' + base
            predictors = [base + '_pre', 'brainseg_pre_cm3']
        cols = ['xtc_group', raw, 'vwrec_delta'] + predictors
        d = data[cols].dropna().copy()
        # Numerical sorting makes resampling invariant to anonymized IDs and CSV row order.
        d = d.sort_values(predictors + [raw, 'vwrec_delta'], kind='stable').reset_index(drop=True)
        a = d[[raw, 'vwrec_delta'] + predictors].to_numpy(float)
        naive = np.flatnonzero(d.xtc_group.eq('XTC-naive'))
        users = np.flatnonzero(d.xtc_group.eq('XTC users'))
        nn = len(naive)
        if min(nn, len(users)) < 3:
            raise ValueError('At least three complete participants per group are required')

        def estimate(idx):
            z = a[idx]
            X = np.column_stack([np.ones(len(z)), z[:, 2:]])
            residual = z[:, 0] - X @ np.linalg.lstsq(X, z[:, 0], rcond=None)[0]
            rn = _rho(residual[:nn], z[:nn, 1])
            ru = _rho(residual[nn:], z[nn:, 1])
            return rn, ru, rn - ru

        original = estimate(np.r_[naive, users])
        rng = np.random.default_rng(seed)
        samples = np.empty((n_boot, 3))
        for i in range(n_boot):
            samples[i] = estimate(np.r_[rng.choice(naive, nn, replace=True),
                                         rng.choice(users, len(users), replace=True)])
            iterations.append({'outcome': outcome, 'iteration': i + 1,
                               'rho_naive': samples[i, 0], 'rho_users': samples[i, 1],
                               'difference_naive_minus_users': samples[i, 2]})
        valid = np.isfinite(samples[:, 2])
        if valid.sum() < 0.99 * n_boot:
            raise ValueError('Too many undefined bootstrap correlations')
        lo, hi = np.quantile(samples[valid, 2], [.025, .975])
        rows.append({'outcome': outcome, 'N_naive': nn, 'N_users': len(users),
                     'rho_naive': original[0], 'rho_users': original[1],
                     'difference_naive_minus_users': original[2], 'ci_low': lo, 'ci_high': hi,
                     'n_bootstrap': n_boot, 'n_valid': int(valid.sum()), 'seed': seed,
                     'ci_method': 'unadjusted percentile 95%', 'exploratory': True})
    fields = ['outcome','N_naive','N_users','rho_naive','rho_users',
              'difference_naive_minus_users','ci_low','ci_high','n_bootstrap','n_valid',
              'seed','ci_method','exploratory']
    pd.DataFrame(rows, columns=fields).to_csv(output_dir / 'selected_correlation_group_bootstrap.csv', index=False)
    pd.DataFrame(iterations, columns=['outcome','iteration','rho_naive','rho_users',
                                    'difference_naive_minus_users']).to_csv(
        output_dir / 'selected_correlation_group_bootstrap_iterations.csv', index=False)
    (output_dir / 'bootstrap_selection.json').write_text(json.dumps({
        'rule': 'Adjusted correlation BH q < alpha in either XTC group; five outcomes per group',
        'alpha': alpha, 'selected_outcomes': selected, 'n_bootstrap': n_boot, 'seed': seed,
        'residualization': 'Refitted on pooled stratified resample in every iteration',
        'intervals': 'Unadjusted, exploratory; do not account for outcome selection',
        'row_order': 'Stable numerical sort, independent of subject identifiers',
    }, indent=2))
    print(f'Exploratory bootstrap completed for {len(selected)} selected outcome(s): {selected}')
