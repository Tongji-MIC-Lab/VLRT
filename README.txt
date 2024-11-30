## Source codes of VLRT

Requirements:
- Python 3.6
- pytorch = 1.4.0
- Other python packages: nltk, pycocotools, pyyaml, easydict, datasets, boto3, ftfy, regex, tqdm

## Data preparation
### msrvtt
For the convenience, you can also download the splits and captions by,
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
### msvd
For the convenience, you can also download them by,
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msvd_data.zip
### activitynet
Download the activitynet datasets ([Download link]: http://activity-net.org/download.html)
### charades
Download the charades datasets ([Download link]: https://prior.allenai.org/projects/charades)

## Compress Video for Speed-up (optional)
python vl-align/preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]

## Weights download
### Download CLIP (ViT-B/32) weight,
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
### Download CLIP (ViT-B/16) weight,
wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

## Training vision-language
bash vl-align/train_vl.sh

## Training captioning 
python main.py

## Testing:
python main_test.py

Note that our experiments are conducted on two 32GB TESLA V100 Cards.