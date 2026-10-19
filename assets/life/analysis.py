# %% 1. 套件、資料與重現設定
from pathlib import Path
import json, hashlib, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import sklearn, statsmodels.api as sm
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA = BASE / 'Life-Expectancy-Data-Updated.csv'
FIG = BASE / 'figures'
FIG.mkdir(exist_ok=True)
SEED = 42
df = pd.read_csv(DATA)
print(f'資料形狀: {df.shape}; 國家: {df.Country.nunique()}; 年份: {df.Year.min()}–{df.Year.max()}')
print(f'缺值: {df.isna().sum().sum()}; 重複列: {df.duplicated().sum()}; 重複國家年度: {df.duplicated(["Country","Year"]).sum()}')
assert not df.isna().any().any()
assert not df.duplicated(['Country', 'Year']).any()
assert ((df.Economy_status_Developed + df.Economy_status_Developing) == 1).all()
df.describe(include='all').T.to_csv(BASE / 'descriptive_statistics.csv', encoding='utf-8-sig')

# %% 2. 資料探索（全資料圖表僅描述，模型設定不依測試結果選擇）
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': '#f6f8fb', 'axes.facecolor': 'white', 'savefig.dpi': 160})
corr = df.corr(numeric_only=True)
corr.to_csv(BASE / 'correlations.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].hist(df.Life_expectancy, bins=25, color='#286c9b', edgecolor='white')
ax[0].set(xlabel='Life expectancy (years)', ylabel='Country-year observations', title='Life expectancy distribution')
ax[1].scatter(df.Schooling, df.Life_expectancy, s=9, alpha=.25, color='#008c95')
b, a = np.polyfit(df.Schooling, df.Life_expectancy, 1)
xx = np.linspace(df.Schooling.min(), df.Schooling.max())
ax[1].plot(xx, a+b*xx, color='#c75235')
ax[1].set(xlabel='Mean years of schooling', ylabel='Life expectancy (years)', title=f'Schooling vs life expectancy (r={corr.loc["Schooling","Life_expectancy"]:.3f})')
fig.tight_layout(); fig.savefig(FIG / '01_eda.png'); plt.close(fig)
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
ax.set_title('Pearson correlation · numerical variables')
fig.colorbar(im, ax=ax, shrink=.7); fig.tight_layout(); fig.savefig(FIG / '02_correlation.png'); plt.close(fig)
print(corr.Life_expectancy.drop('Life_expectancy').sort_values().round(3).to_string())

# %% 3. X/y、80/20 切分及前處理（僅訓練集 fit）
y = df.Life_expectancy
X = df.drop(columns=['Life_expectancy', 'Country', 'Economy_status_Developing'])
numeric = X.select_dtypes(include='number').columns.tolist()
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=.2, random_state=SEED)
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

def make_model(columns, dummy=False):
    nums = [c for c in columns if c != 'Region']
    transformers = [('numeric', StandardScaler(), nums)]
    if 'Region' in columns:
        transformers.append(('region', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), ['Region']))
    return Pipeline([('preprocess', ColumnTransformer(transformers, verbose_feature_names_out=False)),
                     ('regression', DummyRegressor(strategy='mean') if dummy else LinearRegression())])

# 預先指定三個教學比較模型，不做測試集導向的變數篩選。
model_columns = {
    'Mean baseline': list(X.columns),
    'Schooling only': ['Schooling'],
    'Full OLS': list(X.columns),
    'Without mortality': [c for c in X.columns if c not in ['Adult_mortality', 'Infant_deaths', 'Under_five_deaths']]
}
def metrics(actual, predicted):
    return {'R2': r2_score(actual, predicted), 'RMSE_years': float(np.sqrt(mean_squared_error(actual, predicted))),
            'MAE_years': mean_absolute_error(actual, predicted)}

models, rows = {}, []
for name, columns in model_columns.items():
    model = make_model(columns, dummy=name=='Mean baseline')
    model.fit(X_train, y_train)
    models[name] = model
    for split, xx, yy in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
        rows.append({'model': name, 'split': split, 'n': len(yy), **metrics(yy, model.predict(xx))})
performance = pd.DataFrame(rows)
performance.to_csv(BASE / 'model_metrics.csv', index=False, encoding='utf-8-sig')
print(f'Train: {len(train_idx)}, Test: {len(test_idx)}; random_state={SEED}')
print(performance.round(4).to_string(index=False))

# %% 4. 國家分組驗證：訓練集 GroupKFold + 獨立未見國家測試
# CV 僅用原 80% 訓練資料，測試資料不進入 CV。
cv_rows = []
for name in ['Full OLS', 'Without mortality']:
    scores = cross_validate(make_model(model_columns[name]), X_train, y_train,
                            cv=GroupKFold(n_splits=5), groups=df.Country.iloc[train_idx],
                            scoring={'R2':'r2', 'RMSE':'neg_root_mean_squared_error', 'MAE':'neg_mean_absolute_error'})
    for i in range(5):
        cv_rows.append({'model':name, 'fold':i+1, 'R2':scores['test_R2'][i],
                        'RMSE_years':-scores['test_RMSE'][i], 'MAE_years':-scores['test_MAE'][i]})
cv_table = pd.DataFrame(cv_rows)
cv_table.to_csv(BASE / 'group_cv.csv', index=False, encoding='utf-8-sig')
gtrain, gtest = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(X, y, groups=df.Country))
assert set(df.Country.iloc[gtrain]).isdisjoint(set(df.Country.iloc[gtest]))
group_rows=[]
for name in model_columns:
    m = make_model(model_columns[name], dummy=name=='Mean baseline').fit(X.iloc[gtrain], y.iloc[gtrain])
    group_rows.append({'model':name, 'n_train':len(gtrain), 'n_test':len(gtest),
                       **metrics(y.iloc[gtest], m.predict(X.iloc[gtest]))})
group_performance = pd.DataFrame(group_rows)
group_performance.to_csv(BASE / 'unseen_country_metrics.csv', index=False, encoding='utf-8-sig')
print('訓練資料內 5-fold country-group CV (mean / SD):')
print(cv_table.groupby('model')[['R2','RMSE_years','MAE_years']].agg(['mean','std']).round(4).to_string())
print(f'未見國家測試：{df.Country.iloc[gtrain].nunique()} train countries / {df.Country.iloc[gtest].nunique()} test countries')
print(group_performance.round(4).to_string(index=False))

# %% 5. 多元線性迴歸係數、共線性與殘差診斷
full = models['Full OLS']
prep = full.named_steps['preprocess']
names = prep.get_feature_names_out().tolist()
design = pd.DataFrame(prep.transform(X_train), columns=names, index=X_train.index)
design_const = sm.add_constant(design, has_constant='add')
# 同一國家不同年度並非獨立；係數 CI 使用 country-cluster robust SE。
ols = sm.OLS(y_train, design_const).fit(cov_type='cluster', cov_kwds={'groups':df.Country.iloc[train_idx]})
ci = ols.conf_int()
coef = pd.DataFrame({'feature':ols.params.index, 'coefficient':ols.params.values,
                     'cluster_SE':ols.bse.values, 'CI95_low':ci.iloc[:,0].values,
                     'CI95_high':ci.iloc[:,1].values, 'p_value':ols.pvalues.values})
coef['original_unit_coefficient'] = coef.coefficient
scaler = prep.named_transformers_['numeric']
for c, scale in zip(numeric, scaler.scale_):
    coef.loc[coef.feature==c, 'original_unit_coefficient'] /= scale
coef.to_csv(BASE / 'coefficients.csv', index=False, encoding='utf-8-sig')
(BASE / 'ols_summary.txt').write_text(ols.summary().as_text(), encoding='utf-8')
vif = pd.DataFrame({'feature':names, 'VIF':[variance_inflation_factor(design_const.to_numpy(), i+1) for i in range(len(names))]}).sort_values('VIF', ascending=False)
vif.to_csv(BASE / 'vif.csv', index=False, encoding='utf-8-sig')
pred = full.predict(X_test)
resid = y_test.to_numpy() - pred
bp = het_breuschpagan(ols.resid, design_const)
jb = stats.jarque_bera(ols.resid)
predictions = df.iloc[test_idx][['Country','Region','Year','Life_expectancy']].copy()
predictions['Predicted'] = pred
predictions['Residual_actual_minus_predicted'] = resid
predictions['Absolute_error'] = np.abs(resid)
predictions.to_csv(BASE / 'test_predictions.csv', index=False, encoding='utf-8-sig')
print('Adjusted R2 (training):', round(ols.rsquared_adj, 4))
print('Top VIF:'); print(vif.head(8).round(2).to_string(index=False))
print('Breusch-Pagan LM p-value:', bp[1], '; Jarque-Bera p-value:', jb.pvalue)
print(coef[coef.feature.isin(['Adult_mortality','Under_five_deaths','Infant_deaths','Schooling','GDP_per_capita'])].round(4).to_string(index=False))

# %% 6. 圖表與機器可讀摘要
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
ax[0].scatter(y_test, pred, s=12, alpha=.45, color='#167d9a')
lim=[min(y_test.min(), pred.min())-1, max(y_test.max(), pred.max())+1]
ax[0].plot(lim,lim,'--',color='#c75235'); ax[0].set(xlabel='Actual (years)', ylabel='Predicted (years)', title='Holdout: actual vs predicted', xlim=lim, ylim=lim)
ax[1].scatter(pred,resid,s=12,alpha=.45,color='#167d9a'); ax[1].axhline(0,color='#c75235',linestyle='--')
ax[1].set(xlabel='Predicted (years)',ylabel='Actual minus predicted (years)',title='Holdout residuals')
stats.probplot(resid, dist='norm', plot=ax[2]); ax[2].set_title('Holdout residual Q-Q plot')
fig.tight_layout(); fig.savefig(FIG / '03_diagnostics.png'); plt.close(fig)
# 數值欄位（不含二元經濟狀態）的係數：每增加訓練集 1 SD 的預測歲數差。
plot_coef = coef[coef.feature.isin([n for n in numeric if n != 'Economy_status_Developed'])].copy()
plot_coef = plot_coef.reindex(plot_coef.coefficient.abs().sort_values(ascending=False).index).head(10).sort_values('coefficient')
fig, ax = plt.subplots(figsize=(10, 5.8))
ax.barh(plot_coef.feature, plot_coef.coefficient, color=['#c75235' if x<0 else '#167d9a' for x in plot_coef.coefficient])
ax.errorbar(plot_coef.coefficient, plot_coef.feature,
            xerr=np.vstack([plot_coef.coefficient-plot_coef.CI95_low, plot_coef.CI95_high-plot_coef.coefficient]), fmt='none',ecolor='#26374a',capsize=3)
ax.axvline(0,color='#26374a',linewidth=.8); ax.set(xlabel='Years per +1 training SD (95% country-cluster CI)',title='Full OLS · conditional associations, not causal effects')
fig.tight_layout(); fig.savefig(FIG / '04_coefficients.png'); plt.close(fig)
test_perf = performance[performance.split=='Test'].set_index('model')
fig, ax = plt.subplots(figsize=(10, 4.5))
pos=np.arange(len(test_perf)); width=.36
ax.bar(pos-width/2, test_perf.RMSE_years, width, label='Random row holdout', color='#167d9a')
ax.bar(pos+width/2, group_performance.set_index('model').loc[test_perf.index,'RMSE_years'], width, label='Unseen-country holdout', color='#e5a44d')
ax.set_xticks(pos,test_perf.index); ax.set(ylabel='RMSE (years), lower is better',title='Evaluation design changes the question'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / '05_comparison.png'); plt.close(fig)
summary = {
    'rows':len(df), 'countries':df.Country.nunique(), 'train_rows':len(train_idx), 'test_rows':len(test_idx),
    'seed':SEED, 'train_adjusted_R2':float(ols.rsquared_adj),
    'train_design_columns':names, 'train_intercept_standardized':float(ols.params['const']),
    'train_numeric_mean':dict(zip(numeric,scaler.mean_)), 'train_numeric_scale':dict(zip(numeric,scaler.scale_)),
    'region_baseline':str(prep.named_transformers_['region'].categories_[0][0]),
    'BP_LM_pvalue':float(bp[1]), 'JB_pvalue':float(jb.pvalue),
    'data_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
    'versions':{'numpy':np.__version__, 'pandas':pd.__version__, 'sklearn':sklearn.__version__, 'statsmodels':sm.__version__, 'matplotlib':matplotlib.__version__}
}
(BASE / 'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('完成：模型分數、國家分組驗證、係數、預測明細、VIF 與 5 張圖表。')
