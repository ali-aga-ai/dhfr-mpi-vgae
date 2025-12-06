# embed.py will get you protein feature representations, but these feature representations are of size 1280, which is too large for our use case. so we project them down to a  siize of 1024, using a random matrrix, under the assumption that random projection will be learnt by our GNN. we didnt use PCA because for that we need num_samples > num_features, which we dont have here.
import pickle
import numpy as np
import pandas as pd

# Load original vectors
n_components = 1024
random_matrix = np.random.randn(1280, n_components)

# REPLACE with your embedding file name
with open("mutated.pkl", "rb") as f:
    data = pickle.load(f)

all_proteins = np.array(list(data.values()))
print("Original Shape:", all_proteins.shape)

projected_proteins = all_proteins @ random_matrix
print("Projected Shape:", projected_proteins.shape)

# Map keys to reduced vectors
keys = list(data.keys())
reduced_data = {keys[i]: projected_proteins[i] for i in range(len(keys))}

# Load DataFrame
node_feats = pd.read_pickle("new_feature_df_Escherichia_coli.pkl")
print(node_feats.columns)

assigned = 0
missing = 0

# Temporary list for consistent ordering
new_feat_matrix = np.zeros((len(node_feats), n_components), dtype=float)

for idx, row in node_feats.iterrows():
    uid = row.dbid
    if uid in reduced_data:
        new_feat_matrix[idx] = reduced_data[uid]
        assigned += 1
    else:
        missing += 1

# Replace entire column with clean matrix
node_feats["features"] = list(new_feat_matrix)

print("Assigned:", assigned)
print("Missing:", missing)
print("NaN in final matrix:", np.isnan(new_feat_matrix).sum())

# Save only PKL, REPLACE with your desired output file name. 
with open("mutated_new_feature_df_Escherichia_coli.pkl", "wb") as f:
    pickle.dump(node_feats, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved PKL successfully.")
