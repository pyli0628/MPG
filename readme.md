# An effective self-supervised framework for learning expressive molecular global representations to drug discovery

This repository is the official implementation.

> 📋 We provide the code of MPG implementation for pre-training, and fine-tuning on downstream tasks including molecular properties, DDI, and DTI predictions.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Pre-training 

To pre-train the model from scrach, please run the following command to preprocess the data by PSD and AttrMasking:
```train
python pretraining/loader.py 
```
Then, run:
```train
bash pretraining/run_pretraining.sh
```
> 📋 Here, we provide an input example of unlabelled molecular data at `data/pretraining/raw/little_set_smi`. To train the model on 11 M molecules thoroughly, 
please download the [complete pre-training dataset](https://drive.google.com/file/d/1Qdf64BnrUK6RjEuNwzhRMdj6lT-QzZYc/view?usp=sharing) first and put it in the `data/pretraining/raw/` folder.
>

## Fine-tune
> 📋 To finetune the model, please download the pretrained models first and put the model in the `pretrained_model/` folder.


### To fine-tune on molecular propertied prediction, please run:
**Classification tasks**
```finetune
python property/finetune.py --dataset toxcast --lr 0.0001 --input_model_file pretrained_model/MolGNet.pt
```
Note: If you want to run the fine-tune on CPU, please add an argument ```--cpu``` to the command above.
There are more hyper-parameters which can be tuned during finetuning. Please refer to the arguments in ```finetune.py```.

**Regression tasks**
```finetune
python property/finetune_regress.py --dataset esol --lr 0.0001 --input_model_file pretrained_model/MolGNet.pt
```
### To fine-tune on DDI prediction, please run:
**BIOSNAP**
```finetune
python DDI/finetune_snap.py --input_model_file pretrained_model/MolGNet.pt
```
**TWOSIDES**
```finetune
python DDI/finetune_twosides.py --input_model_file pretrained_model/MolGNet.pt
```
### To fine-tune on DDI prediction, please run:
```finetune
python CPI/cross_validate.py --dataset human --input_model_file pretrained_model/MolGNet.pt
```

## Evaluation for reproductivity
Due to the non-deterministic behavior of the function index_select_nd(See [link](https://pytorch.org/docs/stable/notes/randomness.html)) and the randomless of dataset split,
it is hard to exactly reproduce the training process of finetuning. Therefore, we provide the finetuned model and the splitted datasets
for thirteen datasets to guarantee the reproducibility of the experiments. Note: these results are fine-tuned in different hardware environments, resulting in slightly difference from reported statistics in the paper.

**Molecular property prediction**
- [BBBP](https://drive.google.com/drive/folders/1m95c_6F3Df5VzWGgH4k1jjR3NqwXjhj3) 
- [SIDER](https://drive.google.com/drive/folders/1m95c_6F3Df5VzWGgH4k1jjR3NqwXjhj3) 
- [ClinTox](https://drive.google.com/drive/folders/1m95c_6F3Df5VzWGgH4k1jjR3NqwXjhj3) 
- [BACE](https://drive.google.com/drive/folders/1m95c_6F3Df5VzWGgH4k1jjR3NqwXjhj3) 
- [Tox21](https://drive.google.com/drive/folders/1m95c_6F3Df5VzWGgH4k1jjR3NqwXjhj3) 
- [ToxCast](https://drive.google.com/drive/folders/1m95c_6F3Df5VzWGgH4k1jjR3NqwXjhj3) 
- [FreeSolv](https://drive.google.com/drive/folders/1m95c_6F3Df5VzWGgH4k1jjR3NqwXjhj3) 
- [ESOL](https://drive.google.com/drive/folders/1m95c_6F3Df5VzWGgH4k1jjR3NqwXjhj3) 
- [Lipo](https://drive.google.com/drive/folders/1m95c_6F3Df5VzWGgH4k1jjR3NqwXjhj3) 

**DDI prediction**
- [TWOSIDES](https://drive.google.com/drive/folders/19ZkpTnHuxygi4N37kajOcJu5U7OBtMAU) 
- [BIOSNAP](https://drive.google.com/drive/folders/19ZkpTnHuxygi4N37kajOcJu5U7OBtMAU) 

**DTI prediction**
- [human](https://drive.google.com/drive/folders/1S3VLYESORwXLL5q12sAcWy8skEYz59kl) 
- [*elegans*](https://drive.google.com/drive/folders/1S3VLYESORwXLL5q12sAcWy8skEYz59kl) 

We provide the `eval.py` function in `property`, `DDI` and `DTI` folders to reproduce the experiments. 
For example, to evaluate the performance on BBBP dataset, suppose the finetuned model is placed in `finetuned_model/`, please run:
```
python property/eval.py --dataset bbbp --model_dir finetuned_model/property
```
