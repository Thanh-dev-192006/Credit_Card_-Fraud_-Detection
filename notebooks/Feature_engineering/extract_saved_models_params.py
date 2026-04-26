import joblib
import os

model_dir = "d:/AI66A_group_10_Topic_7/models"

def print_model_params(name, pkl_file):
    path = os.path.join(model_dir, pkl_file)
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"--- {name} ---")
        try:
            params = model.get_params()
            print(params)
        except Exception as e:
            print(f"Error getting params: {e}")
    else:
        print(f"File not found: {path}")

print_model_params("LightGBM", "lgb_best.pkl")
print_model_params("XGBoost", "xgb_best.pkl")
print_model_params("CatBoost", "cat_best.pkl")
print_model_params("BRF", "brf_best.pkl")
