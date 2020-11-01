import os
import requests
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from .utils import check_if_exists

from .preprocess_functions import process_image


def download_image(image_url):
    """Downloads image from an url and returns PIL image

    Args:
        image_url (str): url of the desires image

    Returns:
        PIL Image: downloaded image
    """
    resp = requests.get(image_url, stream=True, timeout=5)
    im_bytes = BytesIO(resp.content)
    image = Image.open(im_bytes)
    return image


def build_raw_filepath(file_path, download_folder, unprocessed_folder):
    """Helper function to replace folder on file path

    Args:
        file_path (str): Original file path
        download_folder (str): directory to be replaced
        unprocessed_folder (str): new directory

    Returns:
        str: resulting filepath
    """
    return file_path.replace(f"/{download_folder}/", f"/{unprocessed_folder}/")


def _download_image_multithread_helper(product_tuple):
    """Helper function to run downloads using multithread

    Args:
        product_tuple (tuple): tuple containing ((image_url, file_path), config), where
        image_url is the desired url, file_path where it will be saved and config is the
        configuration dict

    Returns:
        None or error
    """
    (image_url, file_path), config = product_tuple
    try:
        image = download_image(image_url)
        if config["store_unprocessed_images"]:
            raw_path = build_raw_filepath(
                file_path, config["download_folder"], config["unprocessed_folder"]
            )
            check_if_exists(os.path.dirname(raw_path), create=True)
            image.save(raw_path)
        if config["run_preprocess_pipeline"]:
            image = process_image(image, **config["preprocess_config"])
        check_if_exists(os.path.dirname(file_path), create=True)
        image.save(file_path)
        return None
    except IOError:
        return print(f"Error on {file_path}")


def download_image_multithread(download_list, config):
    """Function that receives a download list and generates multiple threads to download in parallel

    Args:
        download_list (list): list with (url, file_path), where url is the desired url and file_path is the save
        location
        config (dict): configuration dict, coming from the config/config.yml

    Returns:
        None
    """
    product_tuple = [(w, config) for w in download_list]
    with ThreadPoolExecutor() as executor:
        executor.map(_download_image_multithread_helper, product_tuple)
    return print("Downloads done")
