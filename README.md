# pictures_downloader

Download and process images pipeline using simple multithreading to make it faster

## How to run it

So on `./config/config.yml`you have all you need to know. Main ones are:

```list
url_source: If you want to query MongoDB or provide a local file with urls and save paths
download_folder: Where images will be saved
run_preprocess_pipeline: If you want to run some processing function on your images before saving them. For instance, here I'm removing faces, resizing and trimming the background
store_unprocessed_image: In case you are running any preprocessing, if you also wants to store the raw image
```

The pipeline uses [Metaflow](https://metaflow.org/) to organize the steps and [multithread](https://docs.python.org/3/library/concurrent.futures.html) to speed things up downloading and processing in parallel.

## Docker

I'm also providing a `Dockerfile` and `docker-compose` in case you want to run this in a container. You just need to

```bash
docker-compose build
```

To build the image and

```bash
docker-compose up
```

To run it.

## Preprocessing function

Here I added some preprocessing steps just as an example. If you don't want or don't need any preprocessing, just leave it blank and you should be fine. In this example I'm using:

- [OpenCV face detection](https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV): As I'm mainly using this for fashion catalog images, I'm assuming there's one face per image and I'm cutting it off
- Resize: Set a maximum size for the largest side
- Trim: Use variance on the edges of the image to remove background.

Basically if does this:
![](https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2020-10-31-download_and_process_pipeline/process_example.jpg)

## Contact
If you have any comments/question, you can finde me [here](https://pauloesampaio.github.io/)