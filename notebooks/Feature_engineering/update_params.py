import json
import re

nb_path = 'Retrain_Graph_ErrorAnalysis.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

brf_params = """n_estimators=1162, max_depth=11, min_samples_leaf=5, min_samples_split=10,
    max_features='sqrt', sampling_strategy='all', replacement=True, random_state=SEED, n_jobs=-1"""

lgb_params = """n_estimators=849, max_depth=9, learning_rate=0.009023826290499453,
    num_leaves=236, min_child_samples=69, subsample=0.9091124239469092,
    colsample_bytree=0.545492087005603, random_state=SEED, n_jobs=-1, verbose=-1"""

xgb_params = """n_estimators=771, max_depth=6, learning_rate=0.02379420156503173,
    subsample=0.8996640755311016, colsample_bytree=0.5738689506596466,
    min_child_weight=13, scale_pos_weight=1.0550726264534553,
    random_state=SEED, n_jobs=-1, tree_method='hist', eval_metric='aucpr', verbosity=0"""

cat_params = """iterations=1983, depth=8, learning_rate=0.006462684586664758,
    l2_leaf_reg=1.027590893752607, subsample=0.6831302982113627,
    colsample_bylevel=0.8616263525510882, min_data_in_leaf=13,
    random_seed=SEED, eval_metric='AUC', verbose=0"""

for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        src = ''.join(cell['source'])
        if 'BalancedRandomForestClassifier(' in src:
            src = re.sub(r'BalancedRandomForestClassifier\([^)]+\)', f'BalancedRandomForestClassifier(\n    {brf_params}\n)', src)
        if 'lgb.LGBMClassifier(' in src:
            src = re.sub(r'lgb\.LGBMClassifier\([^)]+\)', f'lgb.LGBMClassifier(\n    {lgb_params}\n)', src)
        if 'xgb.XGBClassifier(' in src:
            src = re.sub(r'xgb\.XGBClassifier\([^)]+\)', f'xgb.XGBClassifier(\n    {xgb_params}\n)', src)
        if 'CatBoostClassifier(' in src:
            src = re.sub(r'CatBoostClassifier\([^)]+\)', f'CatBoostClassifier(\n    {cat_params}\n)', src)
            
        cell['source'] = [line + '\n' for line in src.split('\n')[:-1]] + [src.split('\n')[-1]]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated parameters in notebook!")
