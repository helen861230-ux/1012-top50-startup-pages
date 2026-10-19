"""Five train-only feature selectors, evaluated on the original shared holdout."""
from itertools import combinations
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, RFECV
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['R&D Spend', 'Administration', 'Marketing Spend',
            'State_Florida', 'State_New York']
METHODS = ['Exhaustive', 'Forward', 'Backward', 'RFECV', 'Lasso']
ALPHAS = np.logspace(-1, 5, 61)
SCORING = 'neg_root_mean_squared_error'


def prepare_data(df):
    """Fixed known encoding; row split exactly matches the reference experiment."""
    X = df[['R&D Spend', 'Administration', 'Marketing Spend']].astype(float).copy()
    X['State_Florida'] = df['State'].eq('Florida').astype(float)
    X['State_New York'] = df['State'].eq('New York').astype(float)
    if not df.State.isin(['California', 'Florida', 'New York']).all():
        raise ValueError('Unexpected state category')
    train_rows, test_rows = train_test_split(df.index.to_numpy(),
                                            test_size=0.2, random_state=0)
    return X, df.Profit, train_rows, test_rows


def linear_pipeline():
    # A fresh scaler is fitted inside each CV fold, including RFECV elimination.
    return Pipeline([('scale', StandardScaler()), ('model', LinearRegression())])


def compare_selection_schemes(df, output_dir):
    """Choose features/hyperparameters on 40 training rows; then score 10 test rows.

    Exhaustive/Forward/Backward evaluate fixed subsets by mean fold RMSE.
    RFECV scores fold-specific elimination subsets for each feature count.
    Lasso tunes alpha and keeps the regularized estimator for final prediction.
    CV scores select a model; they are not nested-CV estimates of generalization.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, y, train_rows, test_rows = prepare_data(df)
    X_train, y_train = X.loc[train_rows], y.loc[train_rows]
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))
    cache, history, fitted = {}, [], {}
    columns = list(X.columns)

    def evaluate(features):
        # Canonical order makes subset caching and tie-breaking deterministic.
        features = tuple(f for f in columns if f in features)
        if features not in cache:
            rmses = -cross_val_score(linear_pipeline(), X_train[list(features)],
                                    y_train, cv=folds, scoring=SCORING, n_jobs=1)
            cache[features] = {'features': features, 'cv_rmse': float(rmses.mean()),
                               'cv_rmse_std': float(rmses.std()),
                               'fold_rmse': rmses.tolist()}
        return cache[features]

    def best(candidates):
        return min(candidates, key=lambda c: (c['cv_rmse'], len(c['features']), c['features']))

    def record(method, stage, candidate):
        row = {'method': method, 'stage': stage, 'n_features': len(candidate['features']),
               'features': ' | '.join(candidate['features']),
               'cv_rmse': candidate['cv_rmse'], 'cv_rmse_std': candidate['cv_rmse_std'],
               'alpha': np.nan, 'chosen': False}
        for i, score in enumerate(candidate['fold_rmse']):
            row[f'fold_{i + 1}_rmse'] = score
        history.append(row)

    # 1. All 2**5 - 1 nonempty subsets, scored using training data only.
    exhaustive = []
    for k in range(1, len(columns) + 1):
        for subset in combinations(columns, k):
            candidate = evaluate(subset)
            exhaustive.append(candidate)
            record('Exhaustive', k, candidate)
    selections = {'Exhaustive': best(exhaustive)}

    # 2. Greedily add one feature at each stage; choose the best size along the path.
    current, forward_path = (), []
    for k in range(1, len(columns) + 1):
        candidates = [evaluate(current + (f,)) for f in columns if f not in current]
        for candidate in candidates:
            record('Forward', k, candidate)
        winner = best(candidates)
        forward_path.append(winner)
        current = winner['features']
    selections['Forward'] = best(forward_path)

    # 3. Start with all features; greedily remove one, then choose the best path size.
    current = tuple(columns)
    backward_path = [evaluate(current)]
    record('Backward', len(columns), backward_path[0])
    while len(current) > 1:
        candidates = [evaluate(tuple(f for f in current if f != removed))
                      for removed in current]
        for candidate in candidates:
            record('Backward', len(current) - 1, candidate)
        winner = best(candidates)
        backward_path.append(winner)
        current = winner['features']
    selections['Backward'] = best(backward_path)

    for method, candidate in selections.items():
        features = list(candidate['features'])
        estimator = linear_pipeline().fit(X_train[features], y_train)
        fitted[method] = (estimator, features, candidate['cv_rmse'],
                          candidate['cv_rmse_std'], None)
        for row in history:
            if row['method'] == method and row['features'] == ' | '.join(features):
                row['chosen'] = True

    # 4. RFE's coefficient magnitudes are on standardized inputs at every fit.
    selector = RFECV(linear_pipeline(), step=1, min_features_to_select=1,
                     cv=folds, scoring=SCORING, n_jobs=1,
                     importance_getter='named_steps.model.coef_').fit(X_train, y_train)
    rf_features = X.columns[selector.support_].tolist()
    rf_results = selector.cv_results_
    rf_index = list(rf_results['n_features']).index(selector.n_features_)
    rf_fold_details = []
    for j, k in enumerate(rf_results['n_features']):
        row = {'method': 'RFECV', 'stage': int(k), 'n_features': int(k),
               # The actual subset can differ across folds; do not label it as fixed.
               'features': '(fold-specific RFE subsets)',
               'cv_rmse': float(-rf_results['mean_test_score'][j]),
               'cv_rmse_std': float(rf_results['std_test_score'][j]),
               'alpha': np.nan, 'chosen': j == rf_index}
        for fold in range(len(folds)):
            row[f'fold_{fold + 1}_rmse'] = float(-rf_results[f'split{fold}_test_score'][j])
            support = rf_results[f'split{fold}_support'][j]
            rf_fold_details.append({'fold': fold + 1, 'n_features': int(k),
                                    'features': X.columns[support].tolist()})
        history.append(row)
    fitted['RFECV'] = (selector.estimator_, rf_features,
                       float(-rf_results['mean_test_score'][rf_index]),
                       float(rf_results['std_test_score'][rf_index]), None)

    # 5. Tune Lasso alpha using the same folds; never refit unregularized OLS afterward.
    lasso = GridSearchCV(
        Pipeline([('scale', StandardScaler()),
                  ('model', Lasso(max_iter=100000, tol=1e-8))]),
        {'model__alpha': ALPHAS}, cv=folds, scoring=SCORING, n_jobs=1,
        refit=True, error_score='raise').fit(X_train, y_train)
    lasso_results = lasso.cv_results_
    for j, alpha in enumerate(ALPHAS):
        row = {'method': 'Lasso', 'stage': j + 1, 'n_features': np.nan,
               'features': '(nonzero support varies by fold)',
               'cv_rmse': float(-lasso_results['mean_test_score'][j]),
               'cv_rmse_std': float(lasso_results['std_test_score'][j]),
               'alpha': float(alpha), 'chosen': j == lasso.best_index_}
        for fold in range(len(folds)):
            row[f'fold_{fold + 1}_rmse'] = float(-lasso_results[f'split{fold}_test_score'][j])
        history.append(row)
    lasso_model = lasso.best_estimator_
    support = lasso_model.named_steps['model'].coef_ != 0
    lasso_features = X.columns[support].tolist()
    fitted['Lasso'] = (lasso_model, lasso_features, float(-lasso.best_score_),
                       float(lasso_results['std_test_score'][lasso.best_index_]),
                       float(lasso.best_params_['model__alpha']))

    # Define every per-count model using training data only. The web chart must
    # follow the selected method rather than reuse the manual reference curve.
    curve_models = []

    def add_curve(method, model, selected, cv_rmse, cv_std, alpha=None):
        curve_models.append({'method': method, 'model': model,
                             'selected': list(selected), 'cv_rmse': cv_rmse,
                             'cv_rmse_std': cv_std, 'alpha': alpha})

    paths = {
        'Exhaustive': [best([c for c in exhaustive if len(c['features']) == k])
                       for k in range(1, len(columns) + 1)],
        'Forward': forward_path,
        'Backward': list(reversed(backward_path)),
    }
    for method, path in paths.items():
        for candidate in path:
            selected = list(candidate['features'])
            model = linear_pipeline().fit(X_train[selected], y_train)
            add_curve(method, model, selected, candidate['cv_rmse'], candidate['cv_rmse_std'])
    for j, k in enumerate(rf_results['n_features']):
        rfe = RFE(linear_pipeline(), n_features_to_select=int(k), step=1,
                  importance_getter='named_steps.model.coef_').fit(X_train, y_train)
        add_curve('RFECV', rfe.estimator_, X.columns[rfe.support_].tolist(),
                  float(-rf_results['mean_test_score'][j]),
                  float(rf_results['std_test_score'][j]))
    lasso_by_count = {}
    for j, alpha in enumerate(ALPHAS):
        model = clone(lasso_model).set_params(model__alpha=float(alpha)).fit(X_train, y_train)
        selected = X.columns[model.named_steps['model'].coef_ != 0].tolist()
        k = len(selected)
        # Show observed nonempty supports, without inventing missing feature counts.
        if k == 0 and lasso_features:
            continue
        cv_rmse = float(-lasso_results['mean_test_score'][j])
        if k not in lasso_by_count or cv_rmse < lasso_by_count[k]['cv_rmse']:
            lasso_by_count[k] = {'model': model, 'selected': selected, 'cv_rmse': cv_rmse,
                                'cv_std': float(lasso_results['std_test_score'][j]),
                                'alpha': float(alpha)}
    for k in sorted(lasso_by_count):
        c = lasso_by_count[k]
        add_curve('Lasso', c['model'], c['selected'], c['cv_rmse'], c['cv_std'], c['alpha'])

    # All selection is finished before the first access to test outcomes.
    rows, predictions, model_details = [], [], {}
    for method in METHODS:
        model, features, cv_rmse, cv_std, alpha = fitted[method]
        # Lasso is fitted on all columns; zero coefficients perform selection.
        input_features = columns if method == 'Lasso' else features
        test_input = X.loc[test_rows, input_features]
        # sklearn RFE fits its final estimator on an ndarray after masking columns.
        pred = model.predict(test_input.to_numpy() if method == 'RFECV' else test_input)
        rows.append({'method': method, 'n_features': len(features),
                     'features': ' | '.join(features), 'cv_rmse': cv_rmse,
                     'cv_rmse_std': cv_std, 'test_rmse': float(np.sqrt(
                         mean_squared_error(y.loc[test_rows], pred))),
                     'test_r_squared': float(r2_score(y.loc[test_rows], pred)),
                     'alpha': alpha})
        predictions.extend({'method': method, 'row_id': int(row),
                            'actual_profit': float(y.loc[row]), 'predicted_profit': float(p),
                            'residual': float(y.loc[row] - p)}
                           for row, p in zip(test_rows, pred))
        scale, reg = model.named_steps['scale'], model.named_steps['model']
        raw_coefs = reg.coef_ / scale.scale_
        model_details[method] = {
            'selected_features': features, 'input_features': input_features, 'alpha': alpha,
            'scaled_coefficients': dict(zip(input_features, reg.coef_.tolist())),
            'scaler_mean': dict(zip(input_features, scale.mean_.tolist())),
            'scaler_scale': dict(zip(input_features, scale.scale_.tolist())),
            'raw_coefficients': dict(zip(input_features, raw_coefs.tolist())),
            'raw_intercept': float(reg.intercept_ - np.dot(raw_coefs, scale.mean_))}
    comparison = pd.DataFrame(rows).sort_values(['cv_rmse', 'n_features', 'method']).reset_index(drop=True)
    comparison.to_csv(out / 'selection_comparison.csv', index=False)
    pd.DataFrame(predictions).to_csv(out / 'selection_predictions.csv', index=False)
    pd.DataFrame(history).to_csv(out / 'selection_cv_history.csv', index=False)
    (out / 'selection_models.json').write_text(json.dumps(model_details, indent=2))
    curve_rows, curve_predictions = [], []
    for c in curve_models:
        method, selected = c['method'], c['selected']
        inputs = columns if method == 'Lasso' else selected
        test_input = X.loc[test_rows, inputs]
        pred = c['model'].predict(test_input.to_numpy() if method == 'RFECV' else test_input)
        chosen = (selected == fitted[method][1] and c['alpha'] == fitted[method][4])
        curve_rows.append({'method': method, 'n_features': len(selected),
                           'features': ' | '.join(selected), 'cv_rmse': c['cv_rmse'],
                           'cv_rmse_std': c['cv_rmse_std'], 'alpha': c['alpha'],
                           'chosen': chosen,
                           'test_rmse': float(np.sqrt(mean_squared_error(y.loc[test_rows], pred))),
                           'test_r_squared': float(r2_score(y.loc[test_rows], pred))})
        curve_predictions.extend({'method': method, 'n_features': len(selected),
                                  'row_id': int(i), 'actual_profit': float(y.loc[i]),
                                  'predicted_profit': float(p)}
                                 for i, p in zip(test_rows, pred))
    pd.DataFrame(curve_rows).to_csv(out / 'selection_feature_curves.csv', index=False)
    pd.DataFrame(curve_predictions).to_csv(out / 'selection_curve_predictions.csv', index=False)
    (out / 'selection_protocol.json').write_text(json.dumps({
        'test_size': 0.2, 'split_random_state': 0, 'row_id_is_zero_based': True,
        'train_rows': train_rows.tolist(), 'test_rows': test_rows.tolist(),
        'cv_n_splits': 5, 'cv_shuffle': True, 'cv_random_state': 42,
        'selection_criterion': 'minimum arithmetic mean of fold RMSE',
        'cv_std_definition': 'population SD across 5 folds; not a confidence interval',
        'scaling': 'StandardScaler fitted inside every CV training fold',
        'dummy_reference': 'California', 'feature_columns': columns,
        'state_selection_unit': 'individual dummy columns, not grouped State',
        'lasso_alpha_grid': ALPHAS.tolist(), 'lasso_support_rule': 'coefficient != 0',
        'final_estimators': 'OLS for first four methods; regularized Lasso for Lasso',
        'folds': [{'fold': i + 1, 'train_rows': train_rows[tr].tolist(),
                   'validation_rows': train_rows[val].tolist()}
                  for i, (tr, val) in enumerate(folds)],
        'rfecv_fold_subsets': rf_fold_details,
        'rfecv_final_ranking': dict(zip(columns, selector.ranking_.tolist())),
        'selection_paths': {'Forward': [list(c['features']) for c in forward_path],
                            'Backward': [list(c['features']) for c in backward_path]},
        'feature_curve_rules': {
            'Exhaustive': 'Lowest training-CV RMSE subset for each count.',
            'Forward': 'One subset per count on the greedy addition path.',
            'Backward': 'One subset per count on the greedy deletion path.',
            'RFECV': 'Training-only full RFE path; CV scores from fold-specific paths.',
            'Lasso': 'Group training-fit nonzero support counts; pick lowest CV RMSE alpha '
                     'per observed count. Missing counts are not interpolated. '
                     'Zero-support candidates omitted unless the CV-selected model has zero support.',
            'evaluation': 'All per-count models are fixed before test outcomes are accessed. '
                          'Test curves are descriptive and never select a method or count.'},
        'interpretation': 'CV scores tune the selectors, not unbiased nested-CV scores. '
                          'The 10-row holdout was previously inspected in image reproduction; '
                          'do not choose a winning scheme from test scores.'}, indent=2))
    return comparison


def plot_selection_comparison(comparison, output_dir):
    """Display in method order; no test-based method ranking."""
    data = comparison.set_index('method').loc[METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(data.index, data.cv_rmse, yerr=data.cv_rmse_std, capsize=4)
    axes[0].set(title='Selection CV RMSE (5 folds; error bars = SD)', ylabel='RMSE')
    axes[1].bar(data.index, data.test_rmse)
    axes[1].set(title='Shared 10-row test RMSE', ylabel='RMSE')
    for axis in axes:
        axis.tick_params(axis='x', rotation=20)
        axis.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / 'selection_comparison.png', dpi=150)
    return fig
