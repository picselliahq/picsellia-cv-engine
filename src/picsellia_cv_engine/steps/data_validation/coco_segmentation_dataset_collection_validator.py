from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from picsellia_cv_engine.models.dataset.dataset_collection import (
    DatasetCollection,
)
from picsellia_cv_engine.models.steps.data_validation.coco_segmentation_dataset_context_validator import (
    CocoSegmentationDatasetContextValidator,
)
from picsellia_cv_engine.models.steps.data_validation.dataset_collection_validator import (
    DatasetCollectionValidator,
)


@step
def coco_segmentation_dataset_collection_validator(
    dataset_collection: DatasetCollection[CocoDatasetContext],
    fix_annotation: bool = False,
) -> None:
    validator = DatasetCollectionValidator(
        dataset_collection=dataset_collection,
        dataset_context_validator=CocoSegmentationDatasetContextValidator,
    )
    validator.validate(fix_annotation=fix_annotation)
