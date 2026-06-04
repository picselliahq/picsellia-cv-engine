"""Group ProcessingType values by the context family they use."""

from picsellia.types.enums import ProcessingType

DATASET_PROCESSING_TYPES: frozenset[ProcessingType] = frozenset(
    {
        ProcessingType.PRE_ANNOTATION,
        ProcessingType.DATASET_VERSION_CREATION,
        ProcessingType.DATA_AUGMENTATION,
        ProcessingType.AUTO_ANNOTATION,
    }
)

DATALAKE_PROCESSING_TYPES: frozenset[ProcessingType] = frozenset(
    {
        ProcessingType.DATA_AUTO_TAGGING,
        ProcessingType.AUTO_TAGGING,
    }
)

MODEL_PROCESSING_TYPES: frozenset[ProcessingType] = frozenset(
    {
        ProcessingType.MODEL_CONVERSION,
        ProcessingType.MODEL_COMPRESSION,
    }
)

PRE_ANNOTATION_LIKE_TYPES: frozenset[ProcessingType] = frozenset(
    {
        ProcessingType.PRE_ANNOTATION,
        ProcessingType.AUTO_ANNOTATION,
    }
)

DATASET_VERSION_OUTPUT_TYPES: frozenset[ProcessingType] = frozenset(
    {
        ProcessingType.DATASET_VERSION_CREATION,
        ProcessingType.DATA_AUGMENTATION,
    }
)


def is_dataset_processing(processing_type: ProcessingType) -> bool:
    return processing_type in DATASET_PROCESSING_TYPES


def is_datalake_processing(processing_type: ProcessingType) -> bool:
    return processing_type in DATALAKE_PROCESSING_TYPES


def is_model_processing(processing_type: ProcessingType) -> bool:
    return processing_type in MODEL_PROCESSING_TYPES
