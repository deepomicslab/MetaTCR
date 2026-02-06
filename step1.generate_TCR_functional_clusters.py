import time
from metatcr.encoder.tcr2vec_encoder import seqlist2ebd, load_tcr2vec
from metatcr.rep2vec import kmeans_clustering, kmeans_traverse_k
from metatcr.utils.utils import save_pk, load_pkfile
from sklearn.cluster import AgglomerativeClustering
import configargparse
import os
import numpy as np
import torch
import random

random.seed(1)

parser = configargparse.ArgumentParser()
parser.add_argument('--database_file', type=str, default='./data/TCR_reference_database.full_legnth.txt', help='Tcr list as reference database')
parser.add_argument('--out_dir', type=str, default='./data/processed_data', help='Output directory for processed data')
parser.add_argument('--primary_k', type=int, default=1024, help='Number of clusters for k-means clustering')
parser.add_argument('--pretrained_model', type=str, default='./pretrained_models/TCR2vec_120', help='Path to the pretrained TCR2vec model')
parser.add_argument('--functional_k', type=int, default=96, help='Number of clusters for the final clustering (functional TCR cluster num)')

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

print("Running with the following parameters:")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print("#######################################")

## read database and transfer to list
with open(args.database_file, 'r') as f:
    lines = f.readlines()
tcrs = [line.rstrip('\n') for line in lines]
print("number of reference TCRs in database:", len(lines))

## load TCR2vec model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path_to_TCR2vec = args.pretrained_model
emb_model = load_tcr2vec(path_to_TCR2vec, device)


print("Encoding TCRs from database...")
X = seqlist2ebd(tcrs, emb_model)
print("X shape:", X.shape)

## ====== Or, skip encoding and load the precomputed embeddings
# X = load_pkfile("./data/all_ref_embedding.pk")

print("Start clustering")
primary_k = args.primary_k
labels, centroids, last_iteration_stat = kmeans_clustering(X, primary_k)

print("clustering finished")
save_pk(os.path.join(out_dir, str(primary_k) + '_primary_labels.pk'), labels)
save_pk(os.path.join(out_dir, str(primary_k) + '_primary_centroids.pk'), centroids)
print("Saved primary clustering results.")

# ## ======= Or, skip clustering and load the precomputed labels and centroids
# labels = load_pkfile(os.path.join(args.out_dir, f"1024_primary_labels.pk"))
# centroids = load_pkfile(os.path.join(args.out_dir, f"1024_primary_centroids.pk"))

spectral_clust = SpectralClustering(n_clusters=functional_k, affinity='rbf', gamma=0.1)
best_labels = spectral_clust.fit_predict(centroids)

best_centroids = np.array([centroids[best_labels == i].mean(axis=0) for i in range(functional_k)])
print("Computed best_centroids shape:", best_centroids.shape)

save_pk(os.path.join(args.out_dir, f"{functional_k}_best_labels.pk"), best_labels)
save_pk(os.path.join(args.out_dir, f"{functional_k}_best_centroids.pk"), best_centroids)
print("Saved spectral clustering results.")
