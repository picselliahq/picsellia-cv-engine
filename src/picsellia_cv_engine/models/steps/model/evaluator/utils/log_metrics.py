import pandas as pd
from picsellia import Experiment
from picsellia.types.enums import LogType


def log_csv_metrics_to_experiment(experiment: Experiment, csv_path: str) -> None:
    """
    Upload COCO evaluation metrics from a CSV file to a Picsellia experiment.

    Args:
        experiment (Experiment): Picsellia experiment object.
        csv_path (str): Path to the CSV file containing evaluation metrics.
    """
    df = pd.read_csv(csv_path).round(3)

    for _, row in df.iterrows():
        category = row["Category"]
        metrics_dict = row.drop("Category").to_dict()
        log_name = f"test/{category}-metrics"
        experiment.log(name=log_name, data=metrics_dict, type=LogType.TABLE)


def upload_metrics_to_picsellia(
    experiment: Experiment,
    csv_path: str,
) -> None:
    """
    Upload COCO evaluation metrics to a Picsellia experiment.

    Args:
        experiment (Experiment): Picsellia experiment object.
        csv_path (str): Path to the CSV file.
    """
    log_csv_metrics_to_experiment(experiment=experiment, csv_path=csv_path)
