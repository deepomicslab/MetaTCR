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


def save_reference_embeddings(X, emb_dir, shard_size=250000, store_dtype=np.float16):
    """Optional intermediate step: cache the encoded reference database as sharded
    ``.npy`` files (+ a manifest), matching ``data/reference_database/reference_embeddings/``.

    Stored as float16 to roughly halve the on-disk size (and far smaller than a single
    ``.pk``); row i of the concatenated shards is the embedding of the i-th database TCR.
    """
    os.makedirs(emb_dir, exist_ok=True)
    n, d = X.shape
    cols = ["shard_id", "start_row", "end_row", "n_rows", "shard_file", "shape", "dtype"]
    rows = []
    for shard_id, start in enumerate(range(0, n, shard_size)):
        end = min(start + shard_size, n)
        shard_file = f"shard_{shard_id:04d}_{start}_{end}.npy"
        np.save(os.path.join(emb_dir, shard_file), X[start:end].astype(store_dtype))
        rows.append({"shard_id": shard_id, "start_row": start, "end_row": end,
                     "n_rows": end - start, "shard_file": shard_file,
                     "shape": f"{end - start}x{d}", "dtype": np.dtype(store_dtype).name})
    with open(os.path.join(emb_dir, "reference_embedding_manifest.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"Saved {len(rows)} embedding shards ({n} x {d}, {np.dtype(store_dtype).name}) to {emb_dir}")


def load_reference_embeddings(emb_dir, dtype=np.float32):
    """Load the sharded reference embeddings back into one ``(n_rows, dim)`` matrix.

    Reads ``reference_embedding_manifest.tsv`` and concatenates the shards in ``shard_id``
    order, so the result aligns 1:1 with the reference database file. Compatible with the
    shards under ``data/reference_database/reference_embeddings/``.
    """
    manifest = os.path.join(emb_dir, "reference_embedding_manifest.tsv")
    with open(manifest) as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [dict(zip(header, ln.rstrip("\n").split("\t"))) for ln in f]
    rows.sort(key=lambda r: int(r["shard_id"]))
    n = sum(int(r["n_rows"]) for r in rows)
    d = int(rows[0]["shape"].split("x")[1])
    X = np.empty((n, d), dtype=dtype)
    pos = 0
    for r in rows:
        a = np.load(os.path.join(emb_dir, r["shard_file"]))
        X[pos:pos + a.shape[0]] = a.astype(dtype)
        pos += a.shape[0]
    print(f"Loaded reference embeddings {X.shape} from {len(rows)} shards in {emb_dir}")
    return X


parser = configargparse.ArgumentParser()
parser.add_argument('--database_file', type=str, default='./data/database/MetaTCR_reference_fullseq.txt', help='Tcr list as reference database')
parser.add_argument('--out_dir', type=str, default='./data/processed_data', help='Output directory for processed data')
parser.add_argument('--primary_k', type=int, default=1024, help='Number of clusters for k-means clustering')
parser.add_argument('--tcr2vec_path', type=str, default='./pretrained_models/TCR2vec_120', help='Path to the pretrained TCR2vec model')
parser.add_argument('--functional_k', type=int, default=128, help='Number of clusters for the final (functional TCR) clustering. Default 128 (recommended setting).')
parser.add_argument('--cluster_mode', type=str, default='spectral', choices=['spectral', 'ward'],
                    help="Functional clustering on the primary centroids. 'spectral' (default, recommended) = "
                         "nearest-neighbors spectral clustering; 'ward' = agglomerative ward (optional alternative).")
parser.add_argument('--spectral_nn', type=int, default=10, help='n_neighbors for the spectral nearest-neighbors affinity (cluster_mode=spectral).')
parser.add_argument('--spectral_seed', type=int, default=0, help='random_state for spectral clustering (cluster_mode=spectral). Default 0 = the recommended seed0 mapping.')
parser.add_argument('--emb_dir', type=str, default='./data/reference_database/reference_embeddings',
                    help='Directory of the encoded reference database (sharded float16 .npy + manifest). '
                         'Used by --save_embeddings / --load_embeddings.')
parser.add_argument('--save_embeddings', action='store_true',
                    help='Optional intermediate step: after encoding, cache the embeddings as sharded .npy '
                         'in --emb_dir (float16 npy, smaller than a single .pk).')
parser.add_argument('--load_embeddings', action='store_true',
                    help='Skip encoding and load the pre-computed embeddings from --emb_dir '
                         '(shards must align 1:1 with --database_file).')
parser.add_argument('--shard_size', type=int, default=250000,
                    help='Rows per .npy shard when --save_embeddings (default 250000, matching the reference embeddings).')

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

# Official output locations: primary centroids/labels under primary_k{K}/, and the
# primary->functional mapping under spectral_mappings/ (read back by steps 2-4).
primary_dir = os.path.join(args.out_dir, f'primary_k{args.primary_k}')
mapping_dir = os.path.join(args.out_dir, 'spectral_mappings')
os.makedirs(primary_dir, exist_ok=True)
os.makedirs(mapping_dir, exist_ok=True)

print("Running with the following parameters:")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print("#######################################")

## ===== Reference embeddings: encode the database, with an optional .npy cache =====
# The encoded reference database is stored as sharded float16 .npy (+ manifest) under
# --emb_dir (default ./data/reference_database/reference_embeddings). Reuse it with
# --load_embeddings (skips encoding), or write it with --save_embeddings after encoding.
if args.load_embeddings:
    print(f"Loading pre-computed reference embeddings from {args.emb_dir} (skip encoding)...")
    X = load_reference_embeddings(args.emb_dir, dtype=np.float32)
else:
    ## read database and transfer to list
    with open(args.database_file, 'r') as f:
        tcrs = [line.rstrip('\n') for line in f]
    print("number of reference TCRs in database:", len(tcrs))

    ## load TCR2vec model and encode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    emb_model = load_tcr2vec(args.tcr2vec_path, device)
    print("Encoding TCRs from database...")
    X = seqlist2ebd(tcrs, emb_model)
    print("X shape:", X.shape)

    ## Optional intermediate step: cache the encoded database as sharded .npy
    if args.save_embeddings:
        save_reference_embeddings(X, args.emb_dir, shard_size=args.shard_size)

print("Start clustering")
primary_k = args.primary_k
labels, centroids, last_iteration_stat = kmeans_clustering(X, primary_k)

print("clustering finished")
save_pk(os.path.join(primary_dir, str(primary_k) + '_primary_labels.pk'), labels)
save_pk(os.path.join(primary_dir, str(primary_k) + '_primary_centroids.pk'), centroids)
print("Saved primary clustering results.")

# ## ======= Or, skip clustering and load the precomputed labels and centroids
# labels = load_pkfile(os.path.join(primary_dir, f"{primary_k}_primary_labels.pk"))
# centroids = load_pkfile(os.path.join(primary_dir, f"{primary_k}_primary_centroids.pk"))


functional_k = args.functional_k
if args.cluster_mode == 'spectral':
    # Recommended default: nearest-neighbors spectral clustering on the primary centroids.
    from sklearn.cluster import SpectralClustering
    print(f"Start spectral clustering (nearest_neighbors, n_neighbors={args.spectral_nn}, "
          f"random_state={args.spectral_seed}) into {functional_k} functional clusters...")
    clust = SpectralClustering(n_clusters=functional_k, affinity='nearest_neighbors',
                               n_neighbors=args.spectral_nn, random_state=args.spectral_seed, n_jobs=-1)
    best_labels = clust.fit_predict(centroids)
else:
    # Optional alternative: ward agglomerative clustering.
    print(f"Start ward hierarchical clustering into {functional_k} functional clusters...")
    agg_clust = AgglomerativeClustering(n_clusters=functional_k, linkage="ward")
    best_labels = agg_clust.fit_predict(centroids)

# primary->functional mapping (dict {primary_cluster_id: functional_cluster_id}) used by create_meta_matrix
centroid_mapping = {i: int(lab) for i, lab in enumerate(best_labels)}
seed_tag = f"seed{args.spectral_seed}_" if args.cluster_mode == "spectral" else ""
mapping_name = f"centroid_mapping_{args.cluster_mode}_{seed_tag}k{functional_k}.pk"
save_pk(os.path.join(mapping_dir, mapping_name), centroid_mapping)
print(f"Saved functional-cluster mapping to {os.path.join(mapping_dir, mapping_name)}.")

best_centroids = np.array([centroids[best_labels == i].mean(axis=0) for i in range(functional_k)])
print("Computed best_centroids shape:", best_centroids.shape)

save_pk(os.path.join(mapping_dir, f"{functional_k}_best_labels.pk"), best_labels)
save_pk(os.path.join(mapping_dir, f"{functional_k}_best_centroids.pk"), best_centroids)
print("Saved hierarchical clustering results.")