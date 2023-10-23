# ESCM
## Code and data for EMNLP 2023 findings: Exploiting Emotion-Semantic Correlations for Empathetic Response Generation.

## Environment
```code
python==3.10.13
torch==2.0.1
```

## Environment Installation
To run this code, you should：
1. Clone the project from github.
```sh
git clone EmpatheticDialogueGeneration_ESCM
```
2. Enter the project root directory
```sh
cd EmpatheticDialogueGeneration_ESCM/
```
3. Download the data, and put it into the project root directory. [Baidu cloud link](https://pan.baidu.com/s/1lHLkNtBiWYyIgc40dJo6jA?pwd=7uqk) with Code: 7uqk or [Google cloud link](https://drive.google.com/file/d/1pxmXP8sG-gmYp2LyNr4eS6J7QOhsplqP/view?usp=sharing)
4. Install required packages
```sh
pip install -r requirements.txt
```
or
```sh
conda env create -f envs.yml
```
The project mainly referenced the code from [CEM](https://github.com/Sahandfer/CEM) and [SEEK](https://github.com/wlr737/EMNLP2022-SEEK). If there are issues during installation, you can try using the environments from these two projects.
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
```sh
nohup python preprocess_vocab.py --device 4 > vocab.log 2>&1 &
```

## Analyzing the emotion-semantic correlations
```sh
nohup python statistics.py > statistics.log 2>&1 &
```
