import matplotlib.pyplot as plt
import os
import json
import itertools
data = []
data_path = "test_outputs_temperature/json_results"
for i, entry in enumerate(os.scandir(data_path)):  
    if entry.is_file(): 
        with open(entry.path) as f:
            data.append(json.load(f))
data = [list(scores.values()) for scores in (file["local_scores"] for file in data)]
data = list(itertools.chain.from_iterable(data))

bins = [i * 0.1 for i in range(11)]
plt.hist(data, bins=bins, edgecolor='black', color='#69b3a2')

# Formatting the chart via plt
plt.title('Histogram of LDDT Local Scores', fontsize=14)
plt.xlabel('Score Range', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(bins) 
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save or show
plt.savefig('histogram_temp_pred_ref.png')
plt.show()