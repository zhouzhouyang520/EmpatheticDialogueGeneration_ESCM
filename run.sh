#!/bin/sh

# Train
output=escm_train.log 
nohup python main.py --model escm --cuda --device_id 4 --pointer_gen --dep_dim 50 > $output 2>&1 &

# Test
#output=escm_train.log 
#nohup python main.py --model escm --test --model_path save/test/CEM_19999_42.5034 --cuda --device_id 2 --batch_size 72 --pointer_gen --dep_dim 50 > $output 2>&1 &

# Print result
tail -f $output
#tail -f escm_test.log 
#tail -f statistics.log 

## Preprocess 
# Preprocess the dependency tree.
#nohup python data_preprocess_tree.py --data_type context  --device 7 > contxt_dep.log 2>&1 &
#nohup python data_preprocess_tree.py --data_type target --device 4 > target_dep.log 2>&1 &

# Preprocess the vocab
#nohup python preprocess_vocab.py --device 4 > vocab.log 2>&1 &

# Analyzing the emotion-semantic correlations
#nohup python statistics.py > statistics.log 2>&1 & 
