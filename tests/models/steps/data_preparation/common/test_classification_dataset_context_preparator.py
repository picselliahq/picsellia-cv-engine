import os
import tempfile
from collections.abc import Callable

from picsellia.types.enums import InferenceType

from picsellia_cv_engine.enums import DatasetSplitName
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


class TestClassificationDatasetContextPreparator:
    def test_extract_categories(
        self, mock_classification_dataset_context_preparator: Callable
    ):
        classification_dataset_organizer = (
            mock_classification_dataset_context_preparator(
                dataset_metadata=DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TRAIN,
                    dataset_type=InferenceType.CLASSIFICATION,
                )
            )
        )

        extracted_categories_dict = (
            classification_dataset_organizer._extract_categories()
        )
        dataset_categories_list = (
            classification_dataset_organizer.dataset_context.labelmap.keys()
        )
        assert len(extracted_categories_dict) == len(dataset_categories_list)
        for category_name in extracted_categories_dict.values():
            assert category_name in dataset_categories_list

    def test_organizer_creates_category_directories(
        self,
        mock_classification_dataset_context_preparator: Callable,
    ):
        with tempfile.TemporaryDirectory() as destination_path:
            classification_dataset_organizer = (
                mock_classification_dataset_context_preparator(
                    dataset_metadata=DatasetTestMetadata(
                        dataset_split_name=DatasetSplitName.TRAIN,
                        dataset_type=InferenceType.CLASSIFICATION,
                    )
                )
            )
            classification_dataset_organizer.destination_path = destination_path

            assert os.path.exists(
                classification_dataset_organizer.dataset_context.images_dir
            ), (
                f"Image directory does not exist: {classification_dataset_organizer.dataset_context.images_dir}"
            )
            assert (
                len(
                    os.listdir(
                        classification_dataset_organizer.dataset_context.images_dir
                    )
                )
                > 0
            ), (
                f"No images found in directory: {classification_dataset_organizer.dataset_context.images_dir}"
            )

            classification_dataset_organizer.organize()

            for (
                category
            ) in classification_dataset_organizer.dataset_context.labelmap.keys():
                category_dir = os.path.join(destination_path, category)
                print(f"category_dir: {category_dir}")
                assert os.path.isdir(category_dir)

    def test_organizer_copies_images_to_correct_directories(
        self,
        mock_classification_dataset_context_preparator: Callable,
    ):
        with tempfile.TemporaryDirectory() as destination_path:
            classification_dataset_organizer = (
                mock_classification_dataset_context_preparator(
                    dataset_metadata=DatasetTestMetadata(
                        dataset_split_name=DatasetSplitName.TRAIN,
                        dataset_type=InferenceType.CLASSIFICATION,
                    )
                )
            )
            classification_dataset_organizer.destination_path = destination_path

            assert os.path.exists(
                classification_dataset_organizer.dataset_context.images_dir
            ), (
                f"Image directory does not exist: {classification_dataset_organizer.dataset_context.images_dir}"
            )
            assert (
                len(
                    os.listdir(
                        classification_dataset_organizer.dataset_context.images_dir
                    )
                )
                > 0
            ), (
                f"No images found in directory: {classification_dataset_organizer.dataset_context.images_dir}"
            )

            classification_dataset_organizer.organize()

            for (
                image
            ) in classification_dataset_organizer.dataset_context.coco_file.images:
                category_id = next(
                    ann.category_id
                    for ann in classification_dataset_organizer.dataset_context.coco_file.annotations
                    if ann.image_id == image.id
                )
                category_name = next(
                    cat.name
                    for cat in classification_dataset_organizer.dataset_context.coco_file.categories
                    if cat.id == category_id
                )
                expected_path = os.path.join(
                    destination_path, category_name, image.file_name
                )
                assert os.path.exists(expected_path), (
                    f"Image {image.file_name} should have been copied to {expected_path}."
                )

    def test_cleanup_removes_original_images_dir(
        self,
        mock_classification_dataset_context_preparator: Callable,
    ):
        with tempfile.TemporaryDirectory() as destination_path:
            classification_dataset_organizer = (
                mock_classification_dataset_context_preparator(
                    dataset_metadata=DatasetTestMetadata(
                        dataset_split_name=DatasetSplitName.TRAIN,
                        dataset_type=InferenceType.CLASSIFICATION,
                    )
                )
            )
            classification_dataset_organizer.destination_path = destination_path

            # Simulate original images directory
            original_images_dir = (
                classification_dataset_organizer.dataset_context.images_dir
            )
            assert os.path.exists(original_images_dir), (
                f"Original image directory does not exist: {original_images_dir}"
            )
            assert len(os.listdir(original_images_dir)) > 0, (
                f"No images found in original directory: {original_images_dir}"
            )

            # Organize the images (which should include cleaning up)
            classification_dataset_organizer.organize()

            # Assert that the original images directory has been removed
            assert not os.path.exists(original_images_dir), (
                "The original images directory should be removed after organizing."
            )
