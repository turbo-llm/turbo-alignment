## models

## dataset
git clone https://huggingface.co/datasets/passing2961/photochat_plus

pip install img2dataset

img2dataset --url_list=photochat_plus/photochat_plus.json --output_folder=images --thread_count=64 --image_size=256 --url_col=photo_url --input_format=json