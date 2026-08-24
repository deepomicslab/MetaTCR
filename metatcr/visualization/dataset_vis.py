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
    plt.title(type)
    file_path = os.path.join(out_dir, 'UMAP_visualization_{}.pdf'.format(type))
    plt.savefig(file_path, dpi = 900, bbox_inches='tight')
