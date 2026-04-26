import json
import os

nb_path = 'Graph_Features.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

plot_cell = {
 'cell_type': 'code',
 'execution_count': None,
 'metadata': {},
 'outputs': [],
 'source': [
  "import matplotlib.pyplot as plt\n",
  "from sklearn.metrics import precision_recall_curve, average_precision_score\n",
  "import joblib\n",
  "import os\n",
  "\n",
  "graph_probs = joblib.load(os.path.join(DATA_DIR, 'graph_test_probs_retrain.pkl'))\n",
  "base_probs = joblib.load(os.path.join(DATA_DIR, 'base_probs_retrain.pkl'))\n",
  "\n",
  "plt.figure(figsize=(10, 8))\n",
  "colors = {'BRF': 'green', 'LightGBM': 'blue', 'XGBoost': 'red', 'CatBoost': 'purple'}\n",
  "\n",
  "for name in graph_probs.keys():\n",
  "    # Baseline PR-AUC\n",
  "    prec_b, rec_b, _ = precision_recall_curve(y_test, base_probs[name])\n",
  "    auc_b = average_precision_score(y_test, base_probs[name])\n",
  "    plt.plot(rec_b, prec_b, linestyle='--', color=colors[name], alpha=0.5,\n",
  "             label=f'{name} Baseline (PR-AUC={auc_b:.4f})')\n",
  "    \n",
  "    # Graph PR-AUC\n",
  "    prec_g, rec_g, _ = precision_recall_curve(y_test, graph_probs[name])\n",
  "    auc_g = average_precision_score(y_test, graph_probs[name])\n",
  "    plt.plot(rec_g, prec_g, linestyle='-', color=colors[name],\n",
  "             label=f'{name} Graph (PR-AUC={auc_g:.4f})')\n",
  "\n",
  "plt.xlabel('Recall')\n",
  "plt.ylabel('Precision')\n",
  "plt.title('PR-AUC Curve: Baseline vs Graph-Enhanced Models (Test Set)')\n",
  "plt.legend(fontsize=9, loc='lower left')\n",
  "plt.grid(True)\n",
  "plt.tight_layout()\n",
  "plt.show()\n"
 ]
}

nb['cells'].append({
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Overlay PR-AUC Curve (All Models)\n",
    "Kết quả từ Retrain_Graph_ErrorAnalysis.ipynb"
   ]
})
nb['cells'].append(plot_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Graph_Features.ipynb updated!')
