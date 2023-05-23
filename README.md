# PAMT: Parametric Attention Mask in Transformer for Continual Image Captioning


The following details the necessary preparations and steps to reproduce the experimental results by running the code:

## Dependencies
This is the list of python requirements:
```
python==3.8.15
torch==1.7.1
torchvision==0.8.2
numpy==1.23.5
pandas==1.5.2
Pillow==9.3.0
h5py==3.7.0
matplotlib==3.6.2
seaborn==0.12.1
bidict==0.22.0
dacite==1.6.0
nltk==3.7
pycocotools==2.0.6
tqdm==4.64.1
attrs==22.1.0
attr==0.3.2
rouge-score==0.1.2
nlg-eval==2.3
dataclasses==0.6
lmdb==1.3.0
scikit-learn==1.1.3
scipy==1.9.3
```
On a common linux distribution, we can create a working environment for this project following this procedure:
1. Create the conda environment with the provided ```environment.yml``` file:
    ```
    conda env create -f environment.yml
    ```  
2. Activate the created environment and install the necessary packages.
3. Install nlg-eval following the instructions described 
[here](https://github.com/Maluuba/nlg-eval), i.e.:
    ```
    pip install git+https://github.com/Maluuba/nlg-eval.git@master
    conda install click
    nlg-eval --setup
    ```
    
## Replication of experimental results

#### Dataset pre-processing
To pre-process MS-COCO:
1. Change the path to the MS-COCO dataset in the first line of ```coco_settings.py```
2. Extract and process image features by running ```scripts/prepro_feats.py```
3. Run the script ```coco_feats.py```.

To pre-process Flickr30k:
1. Change the path to the Flickr30k dataset in the first line of ```flickr30k_settings.py```
2. Extract and process image features by running ```scripts/prepro_feats.py```
3. Run the script ```flickr30k_feats.py```.

#### Model training
The code for model training is currently being organized and optimized.

#### Model evaluation
We provide the trained results in the zip package, which can be used directly for testing. For MS-COCO, the trained model is ```models/transformer_ratt_hard_wholemax_unfreeze/new_smax_700_lambda_5000_epoch_15_1024_step_1```, and for Flickr30k, the trained model is ```models/flickr_ratt/smax_1000_lambda_5000_epoch_50_1024_step_1_warmup_2000```.
The script used for testing is ```eval.py```.
