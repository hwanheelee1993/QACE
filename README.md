# QACE
This repository provides an image captioning metric from our EMNLP-Findings 2021 paper [QACE: Asking Questions to Evaluate an Image Caption
](https://arxiv.org/abs/2108.12560).


## 1) Visual-T5 - Abstractive VQA model
### 0. Detection Feature Extraction
Refer to https://github.com/hwanheelee1993/BUTD-UNITER-NLVR2

### 1. Install Requirements
python 3.6.6\
pip install -r requirements.txt

### 2. Pretrained model download
https://vqamodel.s3.us-east-2.amazonaws.com/t5vqa/ckpt.zip

unzip the file to "ckpt"

### 3. Run Demo
Refer to demo.ipynb

## 2) Computing QACE
Refer to qace_demo.ipynb

Answer similarity computation code will be updated soon. (e.g. using BERTScore)

## Reference
```
@misc{lee2021qace,
      title={QACE: Asking Questions to Evaluate an Image Caption}, 
      author={Hwanhee Lee and Thomas Scialom and Seunghyun Yoon and Franck Dernoncourt and Kyomin Jung},
      year={2021},
      eprint={2108.12560},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
