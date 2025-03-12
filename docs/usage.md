# 🛠 **Creating a custom processing pipeline**

If you want to create a **custom dataset processing pipeline**, `pipeline-cli` simplifies the setup by generating a structured template with all the necessary scripts.

This guide walks you through:

1. Generating a pipeline template
2. Modifying the processing logic
3. Setting up dependencies
4. Testing locally
5. Deploying to Picsellia

---

## **Step 1: Generate a pipeline template**

To create a new pipeline, run:

```sh
pipeline-cli init my_custom_pipeline
```

This will:

- Create a directory `my_custom_pipeline/`
- Generate pre-filled scripts
- Register the pipeline in TinyDB for easy management

Inside `my_custom_pipeline/`, you'll find:
```
my_custom_pipeline/
│── picsellia_pipeline.py
│── local_pipeline.py
│── process_dataset.py
│── Dockerfile
│── requirements.txt
```

- `picsellia_pipeline.py`: Defines the pipeline when running on Picsellia
- `local_pipeline.py`: Used for local testing
- `process_dataset.py`: Modify this file to define how your dataset is processed
- `Dockerfile`: Used to package the pipeline
- `requirements.txt`: Add dependencies here

## **Step 2: Modify `process_dataset.py`**

In your pipeline, the `process_dataset.py` file is where you define how your dataset is processed.
You need to modify the `process_images` function to apply transformations, augmentations, or filtering.

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


## **Step 3: Define dependencies**

No need to create a virtual environment manually! 🎉
When you run the test command, it will automatically create a `.venv` for you and install the dependencies from `requirements.txt`.

Just fill in the `requirements.txt` file with the necessary packages.

To add dependencies, open `requirements.txt` and list them like this:

```txt
opencv-python
torch
```

## **Step 4: Test the pipeline locally**

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

## **Step 5: Deploy the pipeline**

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
