import requests
import pathlib
import os
from PIL import Image
import numpy as np
import cv2
from .utils import check_if_exists


def resize_image(image, dest_size=None, max_side=None):
    """Function to resize image to a specified size or so the largest side has a defined size

    Args:
        image (np.array): Image as a numpy array
        dest_size (int tuple, optional): destination size of the image (width, height)
        max_side (int, optional): [description]. Size of the largest side of the image

    Returns:
        np.array: Array representation of the resized image
    """
    pil_image = Image.fromarray(image)
    if dest_size:
        return np.array(pil_image.resize(dest_size, Image.NEAREST))
    elif max_side:
        scale_factor = max(pil_image.size) / max_side
        (width, height) = (
            pil_image.width // scale_factor,
            pil_image.height // scale_factor,
        )
        resized_image = pil_image.resize((int(width), int(height)), Image.NEAREST)
        return np.array(resized_image)
    else:
        return image


def download_face_detection_model(face_detection_model_path, face_detection_model_url):
    """Function to download face detection model and save it to a specified path

    Args:
        face_detection_model_path (str): directory where model should be saved
        face_detection_model_url (list of str): model url to be downloaded

    Returns:
        None
    """
    check_if_exists(face_detection_model_path, create=True)
    for url in face_detection_model_url:
        model_path = os.path.join(face_detection_model_path, url.split("/")[-1])
        if not check_if_exists(model_path):
            r = requests.get(
                url,
                allow_redirects=True,
            )
            with open(model_path, "wb") as f:
                f.write(r.content)
    return print("Models downloaded")


def load_face_detection_model(model_path):
    """Loads opencv face detection model

    Args:
        model_path (str): Path where model is saved

    Returns:
        OpenCV model: model instance
    """
    model_file = None
    model_config = None
    isPath = check_if_exists(model_path)

    if not isPath:
        print(f"{model_path} not found")
        return None
    else:
        for path in pathlib.Path(model_path).rglob("*.pb"):
            model_file = path.absolute().as_posix()

        for path in pathlib.Path(model_path).rglob("*.pbtxt"):
            model_config = path.absolute().as_posix()

        model = cv2.dnn.readNetFromTensorflow(model_file, model_config)
        return model


def find_and_remove_faces(image, model, pad=10):
    """Function using opencv model to detect and remove faces from images. Assumes one face per image.

    Args:
        image (np.array): Image as a np.array
        model (opencv model): Loaded face detection model
        pad (int, optional): Margin in number of pixels to remove head

    Returns:
        np.array: Representation of the image with face removed
    """
    resized_image = cv2.resize(image, (300, 300))
    blob = cv2.dnn.blobFromImage(
        image=resized_image, mean=(104.0, 177.0, 123.0), swapRB=False
    )
    model.setInput(blob)
    detections = model.forward()
    conf_threshold = 0.5
    h, w = image.shape[:2]
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            bboxes.append((x1, y1, x2, y2))
    if bboxes:
        y_limit = bboxes[0][3] + pad
        cropped = image[y_limit:, :]
        return cropped
    else:
        return image


def trim_image(image, var_threshold=250):
    """Function to remove pixels from image borders that have a variance below a given threshold.
    It gets the rightmost and leftmost pixel columns and calculates its variance. If this variance is below
    threshold, assumes is background and removes it. Do the same with top and bottom pixel rows.

    Args:
        image (np.array): Image as a numpy array
        var_threshold (int, optional): Variance threshold

    Returns:
        np.array: Image array
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    var_x = np.var(blurred, axis=0)
    var_y = np.var(blurred, axis=1)

    limits_x = np.where(var_x > var_threshold)[0]
    if limits_x.size <= 1:
        limits_x = [0, image.shape[1]]

    limits_y = np.where(var_y > var_threshold)[0]
    if limits_y.size <= 0:
        limits_y = [0, image.shape[0]]

    lim_left, lim_right = limits_x[0], limits_x[-1]
    lim_top, lim_bottom = limits_y[0], limits_y[-1]
    return image[lim_top:lim_bottom, lim_left:lim_right]


def trim_iterative(image, initial_threshold=250, delta_threshold=25, min_area_pct=0.25):
    """Function that runs the trim image function iteratively. If it trims to much, reduces the
    variance threshold until trims enough or, if threshold reaches zero, returns the original image

    Args:
        image (np.array): Image array
        initial_threshold (int, optional): Starting variance threshold
        delta_threshold (int, optional): Delta variance between iterations
        min_area_pct (float, optional): Minimum resulting area as a percentage of original image area

    Returns:
        np.array: Image array
    """
    threshold = initial_threshold
    original_x = image.shape[1]
    original_y = image.shape[0]
    done = False
    current_iteration = None
    while not done:
        current_iteration = trim_image(image, threshold)
        if (current_iteration.shape[1] / original_x < min_area_pct) or (
            current_iteration.shape[0] / original_y < min_area_pct
        ):
            threshold = threshold - delta_threshold
        elif (current_iteration.shape[1] / original_x == 1.0) or (
            current_iteration.shape[0] / original_y == 1.0
        ):
            threshold = threshold - delta_threshold
        else:
            done = True
        if threshold <= 0:
            done = True
            current_iteration = image
    return current_iteration


def download_preprocess_requirements(**kwargs):
    """If the preprocess functions require any dependency, get these dependencies.
    For instance, in this case, it needs the face detection models to be downloaded

    Returns:
        None
    """
    download_face_detection_model(**kwargs)
    return None


def process_image(
    image,
    face_detection_model_path,
    face_detection_model_url,
    face_detection_pad,
    resize_max_side,
    trim_initial_variance,
    trim_delta_variance,
    trim_min_area_pct,
):
    """Puts all the preprocess functions in a pipeline

    Args:
        image (PIL image): Image as a PIL image
        face_detection_model_path (str): path of face detection model
        face_detection_pad (int): pad used when removing face from image
        resize_max_side (int): Size of the largest side of resulting image
        trim_initial_variance (int): Initial variance threshold of trim function
        trim_delta_variance (int): Delta of variance between iterations of trim function
        trim_min_area_pct (float): Minimum acceptable area as a percentage of original image area

    Returns:
        PIL Image: Image as PIL Image format
    """
    model = load_face_detection_model(face_detection_model_path)
    image_array = np.array(image)
    faceless = find_and_remove_faces(image_array, model, face_detection_pad)
    resized = resize_image(faceless, max_side=resize_max_side)
    trimmed = trim_iterative(
        resized,
        trim_initial_variance,
        trim_delta_variance,
        trim_min_area_pct,
    )
    return Image.fromarray(trimmed)
