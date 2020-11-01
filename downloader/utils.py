import os
import pathlib
import yaml
import json
import pymongo


def query_db(credentials):
    """Function to query mongodb and get desired information.
    This might change depending on your database structure

    Args:
        credentials (dict): Dictionary with mongodb credentials

    Returns:
        list: list with database response
    """
    connector = (
        f"mongodb+srv://{credentials['user']}:{credentials['password']}"
        f"@{credentials['cluster']}.qe0ku.gcp.mongodb.net/{credentials['database']}?retryWrites=true&w=majority"
    )
    client = pymongo.MongoClient(connector)
    db = client[credentials["database"]]
    docs = db[credentials["collection"]].find({})
    database_response = []
    for doc in docs:
        database_response = database_response + [
            {"url": url, "retailer": doc["product_retailer"], "product_id": doc["_id"]}
            for url in doc["product_images"]
        ]
    return database_response


def parse_database_response(database_response, download_dir):
    """Function to parse database response. Again, this is highly dependent on how your database is structured

    Args:
        database_response (list): List of database query response
        download_dir (str): Path to download directory

    Returns:
        List: List with url and save path
    """
    download_list = []
    for response in database_response:
        download_list = download_list + [
            (
                response["url"],
                os.path.join(
                    download_dir,
                    response["retailer"],
                    str(response["product_id"]) + "_" + response["url"].split("/")[-1],
                ),
            ),
        ]
    return download_list


def load_config(config_path="./config/config.yml"):
    """Function to load configuration set on ./config/config.yml

    Args:
        config_path (str, optional): Path to configuration file file

    Returns:
        Dict: Configuration dict
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def load_credentials(credentials_path):
    """Function to load MongoDb credentials

    Args:
        credentials_path (str): Path to credentials file

    Returns:
        Dict: Credentials dict
    """
    with open(credentials_path, "r") as f:
        credentials = json.load(f).get("mongo_db")
    return credentials


def check_if_exists(path_to_check, create=False):
    """Function to check if a path exists and, if desired, create the containing directory.

    Args:
        path_to_check (string): Path to be checked, directory or filename
        create (bool, optional): In case the path doesn't exist, if the containing directory should be created

    Returns:
        None
    """
    if not os.path.exists(path_to_check):
        if create:
            pathlib.Path(path_to_check).mkdir(parents=True, exist_ok=True)
            return print(f"{path_to_check} created")
        else:
            return None
    else:
        return True
