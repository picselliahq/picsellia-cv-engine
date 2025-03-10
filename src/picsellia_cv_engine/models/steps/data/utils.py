import os


def get_destination_path(job_id: str | None) -> str:
    """
    Generates a destination path based on the current working directory and a job ID.

    Args:
        job_id (Optional[str]): The ID of the current job. If None, defaults to "current_job".

    Returns:
        str: The generated file path for the job.
    """
    if not job_id:
        return os.path.join(os.getcwd(), "current_job")
    return os.path.join(os.getcwd(), str(job_id))
