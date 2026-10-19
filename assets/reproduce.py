"""50 Startups: Kaggle OLS analysis and reference feature-count figure."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from feature_selection import (
    compare_selection_schemes, plot_selection_comparison, prepare_data,
)

ROOT = Path(__file__).resolve().parent
# Explicit sequence inferred from the reference image, not an automatic selector.
FEATURE_ORDER = ['R&D Spend', 'Marketing Spend', 'State_New York',
                 'State_Florida', 'Administration']

def format_results_table(results_df):
    """Format reference or selector test results as a GitHub Markdown table."""
    table = results_df.rename(columns={
        'method': 'Method', 'n_features': 'Number of Features',
        'features': 'Selected Features', 'test_rmse': 'RMSE',
        'test_r_squared': 'R-squared', 'R_squared': 'R-squared',
    }).copy()
    columns = ['Number of Features', 'Selected Features', 'RMSE', 'R-squared']
    if 'Method' in table.columns:
        columns.insert(0, 'Method')
    # Raw feature lists contain pipes, which otherwise split Markdown table cells.
    table['Selected Features'] = table['Selected Features'].str.replace(
        ' | ', ', ', regex=False).str.replace('|', r'\|', regex=False)
    formats = tuple('.2f' if c == 'RMSE' else '.6f' if c == 'R-squared' else 'g'
                    for c in columns)
    return tabulate(table[columns], headers='keys', tablefmt='github',
                    showindex=False, floatfmt=formats)

def load_data():
    df = pd.read_csv(ROOT / 'data' / '50_Startups.csv')
    if df.shape != (50, 5) or df.isna().any().any() or df.duplicated().any():
        raise ValueError('Unexpected dataset shape, missing values or duplicates')
    return df

def compare_features(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # California is the reference state; 3 numeric + 2 dummy columns = 5.
    X, y, train_idx, test_idx = prepare_data(df)
    rows, predictions, coefficients = [], [], []
    for k in range(1, 6):
        features = FEATURE_ORDER[:k]
        model = LinearRegression().fit(X.loc[train_idx, features], y.loc[train_idx])
        pred = model.predict(X.loc[test_idx, features])
        rows.append({'n_features': k, 'features': ' | '.join(features),
                     'RMSE': float(np.sqrt(mean_squared_error(y.loc[test_idx], pred))),
                     'R_squared': float(r2_score(y.loc[test_idx], pred))})
        predictions.extend({'row_id': int(i), 'n_features': k,
                            'actual_profit': float(y.loc[i]), 'predicted_profit': float(p),
                            'residual': float(y.loc[i] - p)}
                           for i, p in zip(test_idx, pred))
        coefficients.append({'n_features': k, 'intercept': float(model.intercept_),
                             'coefficients': dict(zip(features, model.coef_.tolist()))})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / 'metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(output_dir / 'predictions.csv', index=False)
    (output_dir / 'coefficients.json').write_text(json.dumps(coefficients, indent=2))
    (output_dir / 'split.json').write_text(json.dumps({
        'test_size': 0.2, 'random_state': 0, 'row_id_is_zero_based': True,
        'train_rows': train_idx.tolist(), 'test_rows': test_idx.tolist(),
        'feature_order': FEATURE_ORDER, 'reference_state': 'California'}, indent=2))
    return metrics

def plot_comparison(metrics, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    axes[0].plot(metrics.n_features, metrics.RMSE, marker='o')
    axes[0].set(title='RMSE by Number of Features', xlabel='Number of Features', ylabel='RMSE')
    axes[1].plot(metrics.n_features, metrics.R_squared, marker='o')
    axes[1].set(title='R-squared by Number of Features', xlabel='Number of Features', ylabel='R-squared')
    fig.tight_layout()
    fig.savefig(Path(output_dir) / 'feature_comparison.png', dpi=150)
    return fig

def original_analysis(df, output_dir):
    """Reproduce central OLS/standardization/influence steps of the source."""
    output_dir = Path(output_dir)
    encoded = pd.get_dummies(df, columns=['State']).astype(float)
    scaler = StandardScaler()
    z = pd.DataFrame(scaler.fit_transform(encoded), columns=encoded.columns)
    # Drop one state to avoid the original full dummy + intercept singularity.
    full_features = ['R&D Spend', 'Administration', 'Marketing Spend',
                     'State_Florida', 'State_New York']
    full = sm.OLS(z.Profit, sm.add_constant(z[full_features])).fit()
    rd = sm.OLS(z.Profit, sm.add_constant(z[['R&D Spend']])).fit()
    rd_marketing = sm.OLS(z.Profit, sm.add_constant(z[['R&D Spend', 'Marketing Spend']])).fit()
    influence = rd.get_influence().cooks_distance[0]
    # Original notebook drops row 49 for its final in-sample OLS analysis.
    final = sm.OLS(z.loc[z.index != 49, 'Profit'],
                   sm.add_constant(z.loc[z.index != 49, ['R&D Spend']])).fit()
    models = {'full': full, 'RD': rd, 'RD_and_Marketing': rd_marketing,
              'RD_without_row_49': final}
    summaries = pd.DataFrame([{'model': name, 'n_obs': int(m.nobs),
                              'R_squared_train': m.rsquared,
                              'adjusted_R_squared': m.rsquared_adj, 'AIC': m.aic}
                             for name, m in models.items()])
    summaries.to_csv(output_dir / 'ols_metrics.csv', index=False)
    (output_dir / 'ols_summary.txt').write_text('\n\n'.join(
        name + '\n' + m.summary().as_text() for name, m in models.items()))
    pd.DataFrame({'row_id': df.index, 'cooks_distance': influence}).to_csv(
        output_dir / 'influence.csv', index=False)
    # Transform a raw R&D input using the training transformation, then unscale Profit.
    rd_idx, profit_idx = encoded.columns.get_loc('R&D Spend'), encoded.columns.get_loc('Profit')
    standardized_rd = (100000 - scaler.mean_[rd_idx]) / scaler.scale_[rd_idx]
    predicted_z = final.predict(np.array([[1., standardized_rd]]))[0]
    example = {'RD_input': 100000, 'predicted_profit': float(
        predicted_z * scaler.scale_[profit_idx] + scaler.mean_[profit_idx])}
    (output_dir / 'example_prediction.json').write_text(json.dumps(example, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    corr = df.select_dtypes('number').corr()
    im = axes[0].imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    axes[0].set_xticks(range(4), corr.columns, rotation=40, ha='right')
    axes[0].set_yticks(range(4), corr.columns)
    for i in range(4):
        for j in range(4): axes[0].text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center')
    axes[0].set_title('Numeric feature correlations')
    sm.qqplot(rd.resid, line='q', ax=axes[1]); axes[1].set_title('R&D model: residual Q-Q')
    axes[2].scatter(rd.fittedvalues, rd.resid)
    axes[2].axhline(0, color='grey', linestyle='--')
    axes[2].set(title='R&D model: residuals', xlabel='Standardized fitted Profit', ylabel='Standardized residual')
    fig.tight_layout(); fig.savefig(output_dir / 'ols_diagnostics.png', dpi=150)
    return summaries, fig, example

def main():
    df = load_data()
    out = ROOT / 'results'
    metrics = compare_features(df, out)
    plot_comparison(metrics, out)
    selection = compare_selection_schemes(df, out)
    plot_selection_comparison(selection, out)
    summaries, _, example = original_analysis(df, out)
    reference_table = format_results_table(metrics)
    results_df = selection
    selection_table = format_results_table(results_df)
    (out / 'feature_results.md').write_text(reference_table + '\n')
    (out / 'selection_results.md').write_text(selection_table + '\n')
    print(reference_table)
    print('\nAutomatic feature selection (sorted by training CV RMSE):\n')
    print(selection_table)
    print('\nOLS (in-sample; separate from test metrics):\n', summaries.to_string(index=False))
    print('\nExample:', example)
    plt.close('all')

if __name__ == '__main__':
    main()
