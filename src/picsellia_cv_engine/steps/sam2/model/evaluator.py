import os

import torch

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, Model
from picsellia_cv_engine.core.contexts import (
    LocalTrainingContext,
    PicselliaTrainingContext,
)
from picsellia_cv_engine.core.services.model.utils import evaluate_model_impl
from picsellia_cv_engine.frameworks.sam2.services.predictor import SAM2ModelPredictor


@step
def evaluate(model: Model, dataset: CocoDataset):
    """
    Étape d'évaluation pour SAM2. Effectue une inférence complète sur le dataset
    et loggue les résultats dans l'expérience Picsellia.
    """
    context: PicselliaTrainingContext | LocalTrainingContext = (
        Pipeline.get_active_context()
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = SAM2ModelPredictor(model=model, device=device)

    # 📂 Récupérer les chemins d’image
    image_paths = predictor.pre_process_dataset(dataset)

    # 🧠 Inférence
    results = predictor.run_inference_on_images(image_paths)

    # 📤 Post-traitement en prédictions Picsellia
    predictions = predictor.post_process_results(image_paths, results, dataset)

    # 📊 Évaluation Picsellia
    evaluate_model_impl(
        context=context,
        picsellia_predictions=predictions,
        inference_type=model.model_version.type,
        assets=dataset.assets,
        output_dir=os.path.join(context.working_dir, "evaluation"),
        training_labelmap=context.experiment.get_log("labelmap").data,
    )

    print("✅ Évaluation SAM2 terminée et logguée.")
