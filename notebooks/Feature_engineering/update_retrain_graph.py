import json
import os

nb_path = 'Retrain_Graph_ErrorAnalysis.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

save_cell = {
 'cell_type': 'code',
 'execution_count': None,
 'metadata': {},
 'outputs': [],
 'source': [
  "import joblib\n",
  "import os\n",
  "\n",
  "out_path_g = os.path.join(DATA_PATH, 'graph_test_probs_retrain.pkl')\n",
  "out_path_b = os.path.join(DATA_PATH, 'base_probs_retrain.pkl')\n",
  "\n",
  "joblib.dump(graph_test_probs, out_path_g)\n",
  "joblib.dump(base_probs, out_path_b)\n",
  "print('Probabilities saved for Graph_Features.ipynb')\n"
 ]
}

nb['cells'].append({
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Save outputs for downstream Graph_Features.ipynb visualization\n"
   ]
})
nb['cells'].append(save_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Appended save commands to Retrain_Graph_ErrorAnalysis.ipynb')
