import argparse

from cvat_sdk import make_client


class CVATDataLoader:
    """Export dataset from CVAT.

    Args:
        dataset_name (str): Local dataset name.
        username (str): Required CVAT username.
        password (str): Required CVAT password.
        host (str): Required CVAT host.
        format_name (str): CVAT annotation format. Defaults to "CVAT for images 1.1".
    """

    def __init__(
        self,
        dataset_name: str,
        username: str,
        password: str,
        host: str,
        format_name: str = "CVAT for images 1.1",
    ) -> None:
        self.dataset_name = dataset_name
        self.username = username
        self.password = password
        self.host = host
        self.format_name = format_name

    def retrieve_job(self, job_id: int) -> str:
        filename = f"{job_id}.zip"
        with make_client(
            self.host, credentials=(self.username, self.password)
        ) as client:
            job = client.jobs.retrieve(job_id)
            job.export_dataset(self.format_name, filename=filename)
        return filename

    def retrieve_task(self, task_id: int) -> str:
        filename = f"{task_id}.zip"
        with make_client(
            self.host, credentials=(self.username, self.password)
        ) as client:
            task = client.tasks.retrieve(task_id)
            task.export_dataset(self.format_name, filename=filename)
        return filename

    def retrieve_project(self, project_id: int) -> str:
        filename = f"{project_id}.zip"
        with make_client(
            self.host, credentials=(self.username, self.password)
        ) as client:
            project = client.tasks.retrieve(project_id)
            project.export_dataset(self.format_name, filename=filename)
        return filename

    def load_dataset(
        self,
        job_ids: list[str] | None,
        task_ids: list[str] | None,
        project_ids: list[str] | None,
    ) -> str:
        file_paths = []
        if job_ids is not None:
            for job_id in job_ids:
                file_paths.append(self.retrieve_job(job_id))
        if task_ids is not None:
            for task_id in task_ids:
                file_paths.append(self.retrieve_task(task_id))
        if project_ids is not None:
            for project_id in project_ids:
                file_paths.append(self.retrieve_project(project_id))
        return file_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export dataset from CVAT.")
    parser.add_argument("--username", type=str, help="Required CVAT username.")
    parser.add_argument("--password", type=str, help="Required CVAT password.")
    parser.add_argument("--host", type=str, help="Required CVAT host.")
    parser.add_argument(
        "--project_id",
        nargs="+",
        type=int,
        help="ID of existing project.",
        default=None,
    )
    parser.add_argument(
        "--task_id", nargs="+", type=int, help="ID of existing task.", default=None
    )
    parser.add_argument(
        "--job_id", nargs="+", type=int, help="ID of existing job.", default=None
    )
    parser.add_argument(
        "--format", type=str, help="Annotation format.", default="CVAT for images 1.1"
    )
    args = parser.parse_args()
    if all([not args.project_id, not args.task_id, not args.job_id]):
        raise ValueError("Project or task or job IDs should be specified.")

    cvat_loader = CVATDataLoader(
        args.dataset_name, args.username, args.password, args.host, args.format
    )
    file_paths = cvat_loader.load_dataset(args.project_id, args.task_id, args.job_id)
