import os

import torch
from picsellia.types.enums import LogType

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset
from picsellia_cv_engine.core.contexts import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.frameworks.clip.model.model import CLIPModel
from picsellia_cv_engine.frameworks.clip.services.evaluator import (
    apply_dbscan_clustering,
    find_best_eps,
    generate_embeddings_from_results,
    reduce_dimensionality_umap,
    save_cluster_images_plot,
    save_clustering_plots,
    save_outliers_images,
)
from picsellia_cv_engine.frameworks.clip.services.predictor import CLIPModelPredictor


@step()
def evaluate(model: CLIPModel, dataset: CocoDataset):
    """
    Step d’évaluation CLIP via embeddings image uniquement.
    Effectue l'inférence avec le modèle CLIP, puis UMAP + clustering DBSCAN + logs.
    """
    context: PicselliaTrainingContext = Pipeline.get_active_context()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model.trained_weights_path:
        raise FileNotFoundError("No trained weights path found in model.")

    loaded_model, loaded_processor = model.load_weights(
        weights_path=model.trained_weights_path,
        repo_id=context.hyperparameters.model_name,
    )
    model.set_loaded_model(loaded_model)
    model.set_loaded_processor(loaded_processor)

    predictor = CLIPModelPredictor(model=model, device=device)

    # 🔍 Récupère les chemins des images
    image_paths = predictor.pre_process_dataset(dataset)

    # 📦 Batch les images seules
    image_batches = predictor.prepare_batches(
        image_paths, batch_size=context.hyperparameters.batch_size
    )

    # 🧠 Inférence sur images
    results = predictor.run_image_inference_on_batches(image_batches)

    # ➕ Génère les embeddings + chemins images depuis les résultats
    embeddings, paths = generate_embeddings_from_results(image_batches, results)

    # 🌐 UMAP
    reduced = reduce_dimensionality_umap(embeddings, n_components=2)

    # 🔍 Clustering adaptatif
    min_samples = 5
    candidate_eps = [0.1, 0.2, 0.3, 0.5, 0.8]
    best_eps = find_best_eps(reduced, candidate_eps)

    if best_eps is None:
        print("⚠️ No cluster found in first pass, retrying with extended eps...")
        best_eps = find_best_eps(reduced, [0.05, 0.15, 0.25, 0.35, 0.6, 1.0])
    if best_eps is None:
        print("⚠️ Still no clusters, fallback to eps=0.3")
        best_eps = 0.3

    labels = apply_dbscan_clustering(
        reduced, dbscan_eps=best_eps, dbscan_min_samples=min_samples
    )

    # 📊 Logging des résultats
    evaluation_dir = os.path.join(model.results_dir, "clip_evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)

    save_clustering_plots(reduced, labels, results_dir=evaluation_dir)
    save_cluster_images_plot(paths, labels, results_dir=evaluation_dir)
    save_outliers_images(paths, labels, results_dir=evaluation_dir)

    for file in os.listdir(evaluation_dir):
        if file.endswith(".png"):
            context.experiment.log(
                name=f"clip-eval/{file}",
                data=os.path.join(evaluation_dir, file),
                type=LogType.IMAGE,
            )

    print("✅ CLIP evaluation completed and visual logs saved.")
