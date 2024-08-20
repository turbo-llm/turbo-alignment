# Multimodal Tutorial

## Getting started
First, let's download a small dataset suitable for the image captioning task:
```bash
git clone https://huggingface.co/datasets/passing2961/photochat_plus
```

## Download the images
To download the images from the PhotoChat dataset, we will use the `img2dataset` tool:
```bash
pip install img2dataset
```

To download images, use the following command:
```bash
img2dataset --url_list=photochat_plus/photochat_plus.json --output_folder=images --thread_count=64 --image_size=256 --url_col=photo_url --input_format=json
```

We also need to convert the HuggingFace dataset to the `turbo-alignment` format using this script:
```bash
poetry run python tutorials/multimodal/create_tutorial_dataset.py
```

## Train your model
Finally, we are ready to train the model. Run this script to perform the training process.
```bash
poetry run python -m turbo_alignment train_multimodal --experiment_settings_path tutorials/multimodal/multimodal.json
```
