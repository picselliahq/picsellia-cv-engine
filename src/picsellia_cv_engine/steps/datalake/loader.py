from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.processing.datalake.picsellia_datalake_processing_context import (
    PicselliaDatalakeProcessingContext,
)
from picsellia_cv_engine.models.data.datalake.datalake_collection import (
    DatalakeCollection,
)
from picsellia_cv_engine.models.data.datalake.datalake_context import DatalakeContext
from picsellia_cv_engine.models.steps.data.utils import get_destination_path


@step
def load_datalake() -> DatalakeContext | DatalakeCollection:
    """
    Loads and prepares data from a Picsellia Datalake.

    This function retrieves **input and output datalakes** from an active **processing job**
    and downloads all associated data (e.g., images). It supports both **single datalake extraction**
    (input only) and **dual datalake extraction** (input & output).

    Usage:
    - Extracts **one or two datalakes** from the active **processing job**.
    - Downloads all associated data and organizes them into a structured object.
    - Ideal for **data processing tasks requiring images from a Datalake**.

    Behavior:
    - If only an **input datalake** is available, it downloads and returns `DatalakeContext`.
    - If both **input and output datalakes** exist, it returns a `DatalakeCollection`,
      allowing access to both datasets.

    Requirements:
    - The **processing job** must have at least one attached datalake.
    - Ensure `job_id` is set in the active **processing context**.
    - Data assets should be **stored in the Picsellia Datalake**.

    Returns:
        - `DatalakeContext`: If only an **input datalake** is available.
        - `DatalakeCollection`: If both **input and output datalakes** exist.

    Example:
    ```python
    from picsellia_cv_engine.steps.data_extraction.processing.datalake import load_datalake

    # Load datalake data from the active processing job
    datalake_data = load_datalake()

    # Check if the function returned a single datalake or a collection
    if isinstance(datalake_data, DatalakeCollection):
        print("Using both input and output datalakes.")
        print(f"Input datalake images: {datalake_data.input.image_dir}")
        print(f"Output datalake images: {datalake_data.output.image_dir}")
    else:
        print("Using only input datalake.")
        print(f"Input datalake images: {datalake_data.image_dir}")
    ```
    """
    context: PicselliaDatalakeProcessingContext = Pipeline.get_active_context()
    input_datalake_context = DatalakeContext(
        datalake_name="input",
        datalake=context.input_datalake,
        destination_path=get_destination_path(context.job_id),
        data_ids=context.data_ids,
        use_id=context.use_id,
    )
    if context.output_datalake:
        output_datalake_context = DatalakeContext(
            datalake_name="output",
            datalake=context.output_datalake,
            destination_path=get_destination_path(context.job_id),
            use_id=context.use_id,
        )
        datalake_collection = DatalakeCollection(
            input_datalake_context=input_datalake_context,
            output_datalake_context=output_datalake_context,
        )
        datalake_collection.download_all()
        return datalake_collection
    else:
        input_datalake_context.download_data(image_dir=input_datalake_context.image_dir)
        return input_datalake_context
