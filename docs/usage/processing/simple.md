# 🛠 **Creating a Custom Dataset Processing Pipeline**

This guide walks you through how to create, customize, test, and deploy an image processing pipeline using the simple template from `pipeline-cli`. The pipeline is designed for dataset processing tasks such as image transformations, augmentations, or filtering.

1. Generating a pipeline template
2. Modifying the processing logic
3. Setting up dependencies
4. Testing locally
5. Deploying to Picsellia

---

## **1. Initialize your pipeline**

To generate a custom processing pipeline project, run:

```sh
pipeline-cli init my_custom_pipeline --type processing --template simple
```

This will create the pipeline under the pipelines/ folder:

```
pipelines/
└── my_custom_pipeline/
    ├── Dockerfile
    ├── requirements.txt
    ├── local_pipeline.py
    ├── picsellia_pipeline.py
    ├── steps.py
    ├── utils/
    │   ├── processing.py
```

- `picsellia_pipeline.py`: Defines the pipeline when running on Picsellia
- `local_pipeline.py`: Used for local testing
- `steps.py`: Contains the processing steps
- `Dockerfile`: Used to package the pipeline
- `requirements.txt`: Add dependencies here
- `utils/processing.py`: Contains utility functions for processing

## **2. Customize your pipeline**

### Dataset Processing Logic: `steps.py`

In your pipeline, the `steps.py` file is where you define how your dataset is processed. You will define processing steps such as image transformations, augmentations, or filtering within this file.

```python
from picsellia_cv_engine import step

@step()
def process_images(input_images_dir: str, input_coco: dict, output_images_dir: str, output_coco: dict, parameters: dict):
    """
    Modify and save processed images, then update the annotations.
    """
    # Your image processing logic here
    ...
    return output_coco
```

The `process_images` function will be responsible for processing the images in `input_images_dir`, applying transformations or augmentations, and saving the processed images to `output_images_dir`. The COCO annotations will also be updated in `output_coco`.

You can add as many steps as needed in this file to structure the processing pipeline.

### **Understanding the inputs parameters**

Your function receives:

- `input_images_dir` → The directory containing the input dataset images.

- `input_coco` → The COCO annotations from the input dataset.

- `parameters` → The processing parameters configured in the pipeline:

    - Defined inside the processing context:

    ```python
    processing_context = {
        "datalake": "default",
        "data_tag": "processed",
    }
    ```

    - If you need extra parameters, add them to `processing_parameters` in your pipeline files (`local_pipeline.py` and `picsellia_pipeline.py`).

    - `datalake` and `data_tag` are mandatory for uploading to Picsellia.

- `output_images_dir` → An empty directory where the processed images must be saved.

- `output_coco` → An empty COCO dictionary where you should store the updated annotations.

### **What you need to modify**

🔹 Process each image and save it in `output_images_dir`

The pipeline will automatically upload all images from this folder.

Example:

```python
processed_img.save(os.path.join(output_images_dir, image_filename))
```

🔹 Update the COCO annotations in `output_coco`

The `output_coco` dictionary has two keys:

- `"images"` → Stores metadata for processed images.
- `"annotations"` → Stores the updated annotations.

Example of adding a processed image to COCO metadata:

```python
new_image_id = len(output_coco["images"])
output_coco["images"].append(
    {
        "id": new_image_id,
        "file_name": image_filename,
        "width": processed_img.width,
        "height": processed_img.height,
    }
)
```

### **Checklist for `process_images`**

- Modify each image (apply augmentations, transformations, etc.).
- Save all processed images to `output_images_dir`.
- Update `output_coco` with the new image metadata.
- Copy and modify annotations inside `output_coco`.
- Return the updated `output_coco` dictionary.


### ** Define dependencies**

No need to create a virtual environment manually! 🎉
When you run the test command, it will automatically create a `.venv` for you and install the dependencies from `requirements.txt`.

Just fill in the `requirements.txt` file with the necessary packages.

To add dependencies, open `requirements.txt` and list them like this:

```txt
opencv-python
torch
```

## **3. Test your pipeline locally**

Run the pipeline test with:

```sh
pipeline-cli test my_custom_pipeline
```

If it's your first time running the test, the CLI will:

- Create a `.venv` in your pipeline folder.
- Install dependencies from `requirements.txt`.
- Run the pipeline with the correct environment.

During the test, you’ll be prompted for:

- A results directory (where output data will be stored)
- The input dataset version ID
- The output dataset name

✅ If everything works correctly, you're ready to deploy!

## **4. Deploy to pipeline**

Once your pipeline works locally, deploy it to Picsellia:

```sh
pipeline-cli deploy my_custom_pipeline
```

This command will:

1. Build the Docker image
2. Push the image to your Docker registry
3. Register the pipeline in Picsellia

Ensure you are logged into Docker before running this command.

After deployment, you can find your pipeline in Picsellia → Processings → Dataset → Private.
