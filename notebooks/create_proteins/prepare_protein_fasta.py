import pickle
import numpy as np
import requests
import pandas as pd
import time
import requests

# this function fetches the fasta protein sequence from their uniprot id. COMMON ISSUES: timeout, increase timeout value if needed
def get_fasta(uid):
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
    for _ in range(3):  # retry
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.text
        except Exception:
            time.sleep(1)
    return None  # failed

# here, we read the  features from a given pkl file. you can change the code to directly feed the uniprots you want
node_feats = pd.read_pickle("./features/mpi_features/feature_df_Escherichia_coli.pkl")

protein_name = []
non_protein = []
protein_seq = []
r = set()
for _, row in node_feats.iterrows():
    class_name = row['class']

    if class_name == 'protein' or (class_name == "other" and row["node"].endswith("ase")): # the endswith "ase" is a temporary heuristic to capture enzymes mis-labelled as "other", BUT isnt exhaustive

        protein_name.append(row['node'])
        uniprot_id = (row['dbid'])
        print("Analysing protein ", row['node'], " with UniProt ID ", uniprot_id)

        fasta = get_fasta(uniprot_id)
        if fasta is None:
            print("No fasta found for ", uniprot_id)
        else: 
            protein_seq.append(fasta)

    else:
        non_protein.append(row['node'])


print(len(protein_seq))

#  the retreived protein sequences need to be processed so as to be in fasta format with single line sequences, which is expected format for our embedding model. (embed.py)
with open("protein_names.fasta", "w") as f:
    for i, item in enumerate(protein_seq):
        try:
            lines = item.strip().splitlines()
            
            # ensure there's at least a header
            if not lines:
                raise ValueError("Empty item at index {}".format(i))
            
            header = lines[0]
            seq = "".join(lines[1:])   # join ALL sequence lines into one long line

            # Write to file
            f.write(header + "\n")
            f.write(seq + "\n")        # only one newline after sequence

        except Exception as e:
            print(f"Skipping item {i} due to error: {e}")
