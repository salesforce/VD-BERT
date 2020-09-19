# VD-BERT
This repo hosts the source code for our work **VD-BERT: Vision and Dialog Transformer with BERT for Visual Dialog (EMNLP-2020)**.
We have released the pre-trained model `data/saved_models/v1.0_from_BERT_e30.bin` on VisDial v1.0 dataset.

## Installation
See the Dockerfile and Yaml file for GCP `GPU4.yaml`.

For bottom-up attention visual features of VisDial v1.0, we provide them on ``data/img_feats1.0/``. If you would like to extract visual features for other images, please refer to [this docker image](https://hub.docker.com/r/airsplay/bottom-up-attention).
We provide the running script on ``data/visual_extract_code`` (which should be used inside the provided bottom-up-attention image).

## Running Scripts

* Pretraining
  ``bash sh/pretrain_v1.0_mlm_nsp_g4.sh``
* Finetune for discriminative
  ``bash sh/finetune_v1.0_disc_g4.sh``
* Finetune for discriminative specifically on dense annotation
  ``bash sh/finetune_v1.0_disc_dense_g4.sh``
* Finetune for generative
  ``bash sh/finetune_v1.0_gen_g4.sh``
* Test for discriminative on validation
  ``bash sh/test_v1.0_disc_val.shh``
* Test for generative on validation
  ``bash sh/test_v1.0_gen_val.sh``
* Test for discriminative on test
  ``bash sh/test_v1.0_disc_test.sh``
  
(*mlm: masked language modeling, nsp: next sentence prediction, disc: discriminative, gen: generative, g4: 4 gpus, dense: dense annotation*)

## Folder Explanation
``vdbert``: store the main training and testing python files, data loader code, metrics and the ensemble code;

``pytorch_pretrained_bert``: mainly from the Huggingface's [pytorch-transformers v0.4.0](https://github.com/huggingface/pytorch-transformers/tree/v0.4.0);
* `modeling.py`: we only revise two classes: `BertForPreTrainingLossMask` and `BertForVisDialGen`;
* `rank_loss.py`: three ranking methods: ListNet, ListMLE, approxNDCG;

``sh``: shell scripts to run the experiments

``pred``: store two json files for best single-model (74.54 NDCG) and ensemble model (75.35 NDCG)

``data``: VisDial json data and image visual features (38G)

``checkpoints/saved_models``: pretrained models and finetuned models for discriminative setting and generative setting, and optimized on the dense annotation.
  
 
