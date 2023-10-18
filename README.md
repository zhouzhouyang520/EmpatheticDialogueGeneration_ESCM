# EmpatheticDialogueGeneration_ESCM
## Code and data for EMNLP 2023 findings: Exploiting Emotion-Semantic Correlations for Empathetic Response Generation.

## Environment Installation
To run this code, you should：
1. Download the data, and put it into the root folder of the project. （Baidu cloud link: https://pan.baidu.com/s/1lHLkNtBiWYyIgc40dJo6jA?pwd=7uqk
Code: 7uqk）(Google cloud link: )
2. Install the environment.
```sh
pip install -r requirements.txt
```
or
```sh
conda env create -f envs.yml
```
3.
## Model Training
```sh
output=escm_train.log
nohup python main.py --model escm --cuda --device_id 4 --pointer_gen --dep_dim 50 > $output 2>&1 &
```
## Model Test
```sh
output=escm_train.log
nohup python main.py --model escm --test --model_path save/test/CEM_19999_42.5034 --cuda --device_id 2 --batch_size 72 --pointer_gen --dep_dim 50 > $output 2>&1 &
```

## Data Preprocess
If you want to preprocess the dependency tree yourself, you should:
1.Preprocess the dependency tree.
```sh
nohup python data_preprocess_tree.py --data_type context  --device 7 > contxt_dep.log 2>&1 &
nohup python data_preprocess_tree.py --data_type target --device 4 > target_dep.log 2>&1 &
```
2.Preprocess the vocab.
nohup python preprocess_vocab.py --device 4 > vocab.log 2>&1 &

## Analyzing the emotion-semantic correlations
```sh
nohup python statistics.py > statistics.log 2>&1 &
```
