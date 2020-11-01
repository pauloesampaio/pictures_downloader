from metaflow import FlowSpec, step


class DownloadingPipeline(FlowSpec):
    @step
    def start(self):
        """
        Load config
        """
        import pathlib
        from downloader.utils import load_config

        self.FILE_DIR = pathlib.Path(__file__).parent.absolute()
        self.config = load_config()
        self.next(self.get_credentials)

    @step
    def get_credentials(self):
        """
        Load the credentials file.

        """
        from downloader.utils import load_credentials

        self.credentials = load_credentials(self.config["credentials_path"])
        self.next(self.get_download_list)

    @step
    def get_download_list(self):
        """
        Get download list from DB or from local file

        """
        import os
        import csv
        from downloader.utils import query_db, parse_database_response

        if self.config["url_source"] == "mongodb":
            print("Using MongoDB")
            db_response = query_db(credentials=self.credentials)
            self.download_list = parse_database_response(
                db_response, os.path.join(self.FILE_DIR, self.config["download_folder"])
            )
        elif self.config["url_source"] == "local":
            print("Using local file")
            self.download_list = []
            with open("download_list.csv", "r") as file:
                my_reader = csv.reader(file, delimiter=",")
                for row in my_reader:
                    self.download_list.append(row)

        print(f"Found {len(self.download_list)} pictures on the database")
        self.next(self.download_preprocess_requirements)

    @step
    def download_preprocess_requirements(self):
        from downloader.preprocess_functions import download_preprocess_requirements

        if self.config["run_preprocess_pipeline"]:
            required_keys = ["face_detection_model_path", "face_detection_model_url"]
            kwargs = {k: self.config["preprocess_config"][k] for k in required_keys}
            download_preprocess_requirements(**kwargs)
        self.next(self.download_pictures)

    @step
    def download_pictures(self):
        """
        Download images using multiprocess to speed it up

        """
        from downloader.downloader_functions import download_image_multithread

        download_image_multithread(self.download_list, self.config)
        self.next(self.end)

    @step
    def end(self):
        """
        All done!

        """
        print("DONE")


if __name__ == "__main__":
    DownloadingPipeline()
