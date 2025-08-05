import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import umap
from sklearn.metrics import silhouette_score


def generate_embeddings_from_results(image_batches, batch_results):
    """
    Combine image paths and embeddings from batched inference results.

    Args:
        image_batches: List of image path batches.
        batch_results: List of inference result batches.

    Returns:
        A tuple of (embeddings array, image path list).
    """
    all_embeddings = []
    all_paths = []
    for images, results in zip(image_batches, batch_results):
        for img_path, result in zip(images, results):
            all_embeddings.append(result["image_embedding"])
            all_paths.append(img_path)
    return np.array(all_embeddings), all_paths


def load_stored_embeddings(file_path):
    """
    Load stored embeddings and image paths from a .npz file.

    Args:
        file_path: Path to the .npz file.

    Returns:
        Tuple of (embeddings array, image paths).
    """
    data = np.load(file_path, allow_pickle=True)
    return data["embeddings"], data["image_paths"]


def reduce_dimensionality_umap(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """
    Reduce embedding dimensionality using UMAP.

    Args:
        embeddings: High-dimensional embeddings.
        n_components: Target number of dimensions.

    Returns:
        UMAP-reduced embeddings.
    """
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reduced_embeddings = reducer.fit_transform(X=embeddings)
    return reduced_embeddings


def apply_dbscan_clustering(
    embeddings: np.ndarray, dbscan_eps: float, dbscan_min_samples: int
) -> np.ndarray:
    """
    Apply DBSCAN clustering on embeddings.

    Args:
        embeddings: 2D array of points to cluster.
        dbscan_eps: Epsilon parameter for DBSCAN.
        dbscan_min_samples: Minimum samples per cluster.

    Returns:
        Array of cluster labels.
    """
    dbscan = sklearn.cluster.DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels = dbscan.fit_predict(embeddings)
    return labels


def save_clustering_plots(
    reduced_embeddings: np.ndarray, cluster_labels: np.ndarray, results_dir: str
):
    """
    Save annotated DBSCAN clustering plot.

    Args:
        reduced_embeddings: 2D UMAP-reduced embeddings.
        cluster_labels: Cluster labels for each point.
        results_dir: Directory to save the plot.
    """
    os.makedirs(results_dir, exist_ok=True)
    unique_clusters = np.unique(cluster_labels)
    plt.figure(figsize=(10, 6))

    for cluster in unique_clusters:
        indices = np.where(cluster_labels == cluster)[0]
        if cluster == -1:
            plt.scatter(
                reduced_embeddings[indices, 0],
                reduced_embeddings[indices, 1],
                c="gray",
                s=10,
                alpha=0.3,
                label="Outliers",
            )
        else:
            plt.scatter(
                reduced_embeddings[indices, 0],
                reduced_embeddings[indices, 1],
                label=f"Cluster {cluster}",
                s=30,
                alpha=0.7,
            )
            cluster_center = np.mean(reduced_embeddings[indices], axis=0)
            plt.text(
                cluster_center[0],
                cluster_center[1],
                str(cluster),
                fontsize=12,
                weight="bold",
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "black"},
            )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("DBSCAN Cluster Visualization")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(
        os.path.join(results_dir, "dbscan_clusters_annotated.png"), bbox_inches="tight"
    )
    plt.close()


def save_cluster_images_plot(
    image_paths,
    cluster_labels,
    results_dir,
    max_images_per_cluster=25,
    grid_size=(5, 5),
):
    """
    Save a grid of images for each cluster.

    Args:
        image_paths: List of image file paths.
        cluster_labels: Cluster label for each image.
        results_dir: Directory to save plots.
        max_images_per_cluster: Maximum number of images per plot.
        grid_size: Size of the plot grid (rows, cols).
    """
    os.makedirs(results_dir, exist_ok=True)
    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        if cluster == -1:
            continue

        indices = np.where(cluster_labels == cluster)[0]
        if len(indices) == 0:
            continue

        selected_indices = random.sample(
            list(indices), min(max_images_per_cluster, len(indices))
        )

        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
        fig.suptitle(f"Cluster {cluster}", fontsize=14)

        for ax, idx in zip(axes.flatten(), selected_indices, strict=False):
            img = cv2.imread(image_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.axis("off")

        for i in range(len(selected_indices), grid_size[0] * grid_size[1]):
            axes.flatten()[i].axis("off")

        plot_path = os.path.join(results_dir, f"cluster_{cluster}_plot.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()


def save_outliers_images(
    image_paths, cluster_labels, results_dir, max_images=25, grid_size=(5, 5)
):
    """
    Save a grid of images classified as outliers.

    Args:
        image_paths: List of image file paths.
        cluster_labels: Cluster label for each image.
        results_dir: Directory to save the output.
        max_images: Maximum number of outlier images to display.
        grid_size: Size of the output grid (rows, cols).
    """
    os.makedirs(results_dir, exist_ok=True)
    outliers_indices = np.where(cluster_labels == -1)[0]
    if len(outliers_indices) == 0:
        return

    num_images = min(len(outliers_indices), max_images)
    selected_indices = random.sample(list(outliers_indices), num_images)

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
    fig.suptitle("DBSCAN Outliers", fontsize=14)

    valid_images = 0
    for ax, idx in zip(axes.flatten(), selected_indices, strict=False):
        image_path = str(image_paths[idx])
        img = cv2.imread(image_path)
        if img is None:
            ax.axis("off")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis("off")
        valid_images += 1

    for i in range(valid_images, grid_size[0] * grid_size[1]):
        axes.flatten()[i].axis("off")

    outliers_path = os.path.join(results_dir, "outliers_images.png")
    plt.savefig(outliers_path, bbox_inches="tight")
    plt.close()


def find_best_eps(reduced, eps_list):
    """
    Find the best epsilon value for DBSCAN using silhouette score.

    Args:
        reduced: 2D array of reduced embeddings.
        eps_list: List of candidate epsilon values.

    Returns:
        The epsilon value with the highest silhouette score.
    """
    best_eps = None
    best_score = -1
    for eps in eps_list:
        db = sklearn.cluster.DBSCAN(eps=eps, min_samples=5).fit(reduced)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 2:
            score = silhouette_score(reduced, labels)
            if score > best_score:
                best_score = score
                best_eps = eps
    return best_eps
