import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import os

from scipy.stats import entropy

def jensen_shannon_divergence(p, q):
    """Calculates the Jensen-Shannon Divergence between two probability distributions."""
    mask = (p != 0) | (q != 0)
    p_filtered = p[mask]
    q_filtered = q[mask]
    m = 0.5 * (p_filtered + q_filtered)
    jsd = 0.5 * (entropy(p_filtered, m) + entropy(q_filtered, m))
    return jsd

def plot_combined_datasets_umap(mtx, setnames, type="cluster_TCR_diversity", out_dir="./results/data_analysis", min_dist=0.1, n_neighbors=30, dim=2):
    """
    Visualizes a combined matrix of multiple datasets using UMAP, coloring points by dataset.
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating UMAP embedding for {mtx.shape[0]} samples...")
    
    # ========================== CRITICAL FIX ==========================
    # The `n_jobs=1` parameter is ESSENTIAL to prevent a multiprocessing
    # error in the underlying 'pynndescent' library. This forces UMAP
    # to run in single-threaded mode, avoiding the bug.
    # ==================================================================
    umap_runner = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=dim,
        random_state=42,
        n_jobs=1
    )

    print("merged mtx:", mtx.shape)
    embedding = umap_runner.fit_transform(mtx.astype(np.float32))

    df = pd.DataFrame(embedding, columns=['UMAP 1', 'UMAP 2'])
    df['Dataset'] = setnames

    s_size = 10 if mtx.shape[0] > 1000 else 30

    # --- Plotting Section ---
    fig, ax = plt.subplots(figsize=(10, 7))

    unique_setnames = sorted(list(set(setnames)))
    num_labels = len(unique_setnames)
    palette = sns.color_palette("tab20", num_labels) if num_labels <= 20 else sns.color_palette("hls", num_labels)
    color_map = dict(zip(unique_setnames, palette))

    sns.scatterplot(
        x='UMAP 1',
        y='UMAP 2',
        hue='Dataset',
        hue_order=unique_setnames,
        palette=color_map,
        data=df,
        legend='full',
        alpha=0.7,
        s=s_size,
        edgecolor='none',
        ax=ax
    )

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title='Dataset')
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    file_path = os.path.join(out_dir, f'UMAP_visualization_{type}.pdf')
    print(f"Saving plot to: {file_path}")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)



def visualize_metavec_inset(mtx, smplist, df_metadata, refdata, id_col, label_col, min_dist=0.2, n_neighbors=20, dim=2, type="TCR diversity", out_dir = "./results/data_analysis"):
    """
    Advanced UMAP visualization that can incorporate metadata and reference data.
    """
    df_metadata.set_index(id_col, inplace=True)

    common_samples = list(set(smplist) & set(df_metadata.index))
    common_indices = [smplist.index(sample) for sample in common_samples]
    mtx = mtx[common_indices, :]
    smplist = [smplist[i] for i in common_indices]

    if isinstance(label_col, str):
        label_col = [label_col]

    labels_mtx = []
    for col in label_col:
        labels_mtx.append([df_metadata.at[sample_id, col] for sample_id in smplist])

    if refdata is None:
        data = mtx
        labels = labels_mtx
        s_size = 30
    else:
        data = np.vstack((mtx, refdata))
        labels_ref = ["Reference"] * refdata.shape[0]
        labels = [labels_m + labels_ref for labels_m in labels_mtx]
        s_size = 20
        smplist = smplist + ["Reference"] * refdata.shape[0]

    embedding = umap.UMAP(min_dist=min_dist,n_neighbors=n_neighbors, n_components=dim, random_state=1, n_jobs=1).fit_transform(data)

    df = pd.DataFrame(embedding, columns=["Umap1", "Umap2"])
    df['sample'] = smplist

    fig, axs = plt.subplots(1, len(label_col), figsize=(15, 5)) if len(label_col) > 1 else plt.subplots(figsize=(10,10))
    if not isinstance(axs, np.ndarray): axs = [axs]

    for i, label in enumerate(labels):
        plt.sca(axs[i])
        df["label"] = label
        if refdata is not None:
            sns.scatterplot(x="Umap1", y="Umap2", hue="label", data=df[df["label"] == "Reference"], palette=["gray"], legend="full", alpha=0.5, s=s_size)
        sns.scatterplot(x="Umap1", y="Umap2", hue="label", data=df[df["label"] == "Unkown"], legend="full", alpha=1, palette=["lightgray"], s=s_size)
        sns.scatterplot(x="Umap1", y="Umap2", hue="label", data=df[np.logical_and(df["label"] != "Reference", df["label"] != "Unkown")], legend="full", alpha=0.8, s=s_size, palette="Set2")
        axs[i].set_title(str(label_col[i]))
        for spine in axs[i].spines.values():
            spine.set_linewidth(1.5)
            
    plt.suptitle("UMAP - datasets: " + type)
    plt.subplots_adjust(top=0.8, bottom=0.1)
    plt.savefig(os.path.join(out_dir, "UMAP_visualization_of_datasets_{}.png".format(type)), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "UMAP_visualization_of_datasets_{}.svg".format(type)), dpi=600, bbox_inches='tight')
    plt.close(fig)

def visualize_metavec(mtx, setnames, min_dist=0.1, n_neighbors=50, dim=2, type = "cluster TCR diversity", out_dir = "./results/data_analysis"):

    """
    Use UMAP to visualize all datasets in mtxs. Color each dataset differently.
    mtxs: a numpy array of shape (n_samples, n_features)
    setnames: a list of dataset names, len(setnames) == mtxs.shape[0]
    dim: dimension of UMAP embedding
    """

    # UMAP embed all data
    embedding = umap.UMAP(min_dist=min_dist, n_neighbors=n_neighbors, n_components=dim, random_state=0).fit_transform(mtx)

    # Make a color palette with a color for each dataset
    # palette = sns.color_palette("tab20", len(setnames))
    unique_setnames = set(setnames)
    num_label = len(unique_setnames)
    if num_label <= 10:
        palette = sns.color_palette("tab10", num_label)
    else:
        palette = sns.color_palette("hls", num_label)


    # Create a dataframe with the embedding and dataset labels
    df = pd.DataFrame(embedding, columns=['Umap1', 'Umap2'])

    df['dataset'] = setnames
    if mtx.shape[0] > 1000:
        s_size = 10
    else:
        s_size = 30

    # Plot the UMAP embedding colored by dataset

    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.scatterplot(x='Umap1', y='Umap2', hue='dataset', palette=palette,
                         data=df, legend='full', alpha=0.5, s=s_size, edgecolor='none')


    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.title('UMAP visualization: ' + type)
    plt.title(type)
    file_path = os.path.join(out_dir, 'UMAP_visualization_{}.pdf'.format(type))
    plt.savefig(file_path, dpi = 900, bbox_inches='tight')




# def inner_dist_mtx(mtx):
#     smp_num = mtx.shape[0]
#     dist_matrix = np.zeros((smp_num, smp_num))
#     for i in range(smp_num):
#         for j in range(i+1, smp_num):
#             ## euclidean distance
#             dist = np.linalg.norm(mtx[i] - mtx[j])
#             dist_matrix[i, j] = dist
#             dist_matrix[j, i] = dist
#     return dist_matrix

# def outer_dist_mtx(mtx1, mtx2):
#     smp_num1 = mtx1.shape[0]
#     smp_num2 = mtx2.shape[0]
#     dist_matrix = np.zeros((smp_num1, smp_num2))
#     for i in range(smp_num1):
#         for j in range(smp_num2):
#             ## euclidean distance
#             dist = np.linalg.norm(mtx1[i] - mtx2[j])
#             dist_matrix[i, j] = dist
#     return dist_matrix

# def get_upper_triangle(dist_matrix):
#     upper_triangle = np.triu(dist_matrix, k=1)
#     result = upper_triangle[upper_triangle != 0].tolist()
#     return result

# def calc_metrics(mtx1, mtx2):
#     ### get the summary of the distance between two matrix
#     mtx1_dist = inner_dist_mtx(mtx1)
#     mtx1_dist = get_upper_triangle(mtx1_dist)
#     mtx2_dist = inner_dist_mtx(mtx2)
#     mtx2_dist = get_upper_triangle(mtx2_dist)
#     mtx1_mtx2_dist = outer_dist_mtx(mtx1, mtx2)

#     ## mean
#     mean_mtx1 = np.mean(mtx1_dist)
#     mean_mtx2 = np.mean(mtx2_dist)
#     mean_mtx1_mtx2 = np.mean(mtx1_mtx2_dist)

#     ## Variance
#     var_mtx1 = np.var(mtx1_dist)
#     var_mtx2 = np.var(mtx2_dist)

#     ## Coefficient of Variation
#     cv_mtx1 = np.std(mtx1_dist) / mean_mtx1 if mean_mtx1 else 0
#     cv_mtx2 = np.std(mtx2_dist) / mean_mtx2 if mean_mtx2 else 0

#     return mean_mtx1, mean_mtx2, mean_mtx1_mtx2, var_mtx1, var_mtx2, cv_mtx1, cv_mtx2

# def mean(num1, num2):
#         return (num1 + num2) / 2
    
# def diff_score(mean_mtx1, mean_mtx2, mean_mtx1_mtx2, var_mtx1, var_mtx2, cv_mtx1, cv_mtx2):
#     alpha = 0.4
#     beta = 0.4
#     gamma = 0.2
#     # score = alpha * (mean_mtx1_mtx2 / max(mean_mtx1, mean_mtx2)) + beta * (max(var_mtx1, var_mtx2) / min(var_mtx1, var_mtx2)) + gamma * (max(cv_mtx1, cv_mtx2) / min(cv_mtx1, cv_mtx2))
#     score = alpha * (mean_mtx1_mtx2 / max(mean_mtx1, mean_mtx2)) + beta * (
#             max(var_mtx1, var_mtx2) / mean(var_mtx1, var_mtx2)) + gamma * (
#                     max(cv_mtx1, cv_mtx2) / mean(cv_mtx1, cv_mtx2))
    
#     return score