import json
import glob

notebooks = [
    "LightGBM_XGBoost.ipynb",
    "Catboost.ipynb",
    "BalancedRandomForestClassifier.ipynb"
]

for nb_name in notebooks:
    with open(nb_name, "r", encoding="utf-8") as f:
        nb = json.load(f)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                for out in cell.get("outputs", []):
                    if out.get("output_type") == "stream" and out.get("name") == "stdout":
                        text = "".join(out.get("text", []))
                        if "Best params:" in text:
                            print(f"--- {nb_name} ---")
                            for line in text.split("\n"):
                                if "Best params:" in line:
                                    print(line)
