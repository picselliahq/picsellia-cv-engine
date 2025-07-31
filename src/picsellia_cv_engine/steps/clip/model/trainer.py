import os

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from picsellia_cv_engine.frameworks.clip.services.trainer import ClipModelTrainer


@step()
def train(model: Model, dataset_collection: DatasetCollection[CocoDataset]):
    """
    Step d'entraînement pour CLIP via le moteur de training Picsellia.
    Cette step utilise BLIP pour générer les captions et entraîne CLIP en ligne de commande.
    """
    context = Pipeline.get_active_context()

    # 📁 Répertoire d'entraînement (où seront stockés JSON + checkpoints)
    output_dir = os.path.join(model.results_dir, "clip_finetuned")
    os.makedirs(output_dir, exist_ok=True)

    # 🚀 Lancer le training
    trainer = ClipModelTrainer(
        context=context,
        model_dir=output_dir,
    )
    trainer.train_model(dataset_collection=dataset_collection)
