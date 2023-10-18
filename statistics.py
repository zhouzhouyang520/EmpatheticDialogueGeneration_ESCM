'''
Biaffine Dependency parser from AllenNLP
'''
import argparse
import json
import os
import re
import sys
import datetime
import time

from tqdm import tqdm
import pickle
import numpy as np

from src.utils.data.loader import load_dataset, load_dep_tree, load_vad
from src.utils.constants import EMO_MAP


def convert(n):
    return str(datetime.timedelta(seconds = n)) 

def wrapper_calc_time(print_log=True):
    """ 
    Compute execute time of function
    :param print_log: print log or not.
    :return:
    """

    def wrapper(func):
        def inner_wrapper(*args, **kwargs):
            start_time = time.time()
            func_re = func(*args, **kwargs)
            run_time = time.time() - start_time
            converted_time = convert(run_time)
            if print_log:
                print(f"{func.__name__} time:", run_time, converted_time)
            return func_re

        return inner_wrapper

    return wrapper


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--data_path', type=str, default='./data/ED/',
                        help='Directory of where ED data held.')
    parser.add_argument('--save_name', type=str, default='dep_tree.p',
                        help='Directory of where ED data held.')
    parser.add_argument('--data_type', type=str, default='context',
                        help='Directory of where ED data held.')
    parser.add_argument('--device', type=int, default=7)
    return parser.parse_args()

def build_emo_map():
    emo_map = {}
    for key in EMO_MAP:
        emo_map[key] = {}
    return emo_map

def emotion_intensity(word, vad):
    v, a, d = vad[0], vad[1], vad[2]
    a = a/2 
    score = np.linalg.norm(np.array([v, a]) - np.array([0.5, 0]))
    #print(word, score)
    return  score > 0.47 #0.06467

@wrapper_calc_time(print_log=True)
def statistics_correlations():
    args = parse_args()

    print("Loading ED dataset...")
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    ed_data_list = [pairs_tra, pairs_val, pairs_tst]
    #ed_data_list = [pairs_tst]

    print("Loading dependency tree...")
    dep_tree_tra, dep_tree_val, dep_tree_tst, tag_set, dep_set = load_dep_tree()
    dep_data_list = [dep_tree_tra, dep_tree_val, dep_tree_tst]
    #dep_data_list = [dep_tree_tst]
    
    print("Loading vad...")
    origin_vad_dict = load_vad()
    print(f"origin_vad_dict length: {len(origin_vad_dict)}")
    vad_dict = dict(filter(lambda x: emotion_intensity(x[0], x[1]), origin_vad_dict.items()))
    # 2580/20007
    #print(f"vad_dict length: {len(vad_dict)}, {list(vad_dict.items())[:20]}")

    total_map = {}
    emo_res_map = build_emo_map() #
    for index, ed_data in enumerate(ed_data_list):
        dep_data = dep_data_list[index] 
        data_length = len(ed_data["context"])
        print(f"index: {index}, data_length: {data_length}")
        for i in range(data_length):
            emotion = ed_data["emotion"][i]
            emotion_context = ed_data["emotion_context"][i]
            dep_context_list = dep_data["context"][i]
            for emo_word in emotion_context: 
                if emo_word not in vad_dict:
                    continue
                print(f"valid emo_word: {emo_word}")
                for dep_context in dep_context_list:
                    tokens = dep_context["tokens"] 
                    tmp_tokens = " ".join(tokens)
                    tags = dep_context["tags"]
                    dependencies = dep_context["dependencies"]
                    predicted_heads = dep_context["predicted_heads"]
                    
                    # Find tail emotion word.
                    res = find_by_emo_tail(emo_word, tokens, tags, dependencies)
                    if res is not None: 
                        correlation_key, words = res[0], res[1]
                        print(f"tokens: {tokens}")
                        print(f"correlation_key: {emo_word}, {correlation_key}, {words}")
                        update_dict(correlation_key, total_map, emo_res_map, emotion)

                    # Find head emotion word.
                    res = find_by_emo_head(emo_word, tokens, tags, dependencies, predicted_heads)
                    if res is not None: 
                        correlation_key, words = res[0], res[1]
                        print(f"tokens: {tokens}")
                        print(f"correlation_key: {emo_word}, {correlation_key}, {words}")
                        update_dict(correlation_key, total_map, emo_res_map, emotion)
            #print("====================================")

    precent_correlation = open(args.data_path + "precent_correlation.csv", "w")
    topk_correlation = open(args.data_path + "topk_correlation.csv", "w")
    print("Sort total_map:")
    sort_write_map(total_map, precent_correlation, topk_correlation, "total", topk=2)
    for key in emo_res_map:
        print(f"emo_res_map: {key}")
        sort_write_map(emo_res_map[key], precent_correlation, topk_correlation, key, topk=2)
        print("======================")
#    print("Save data...")
#    with open(args.data_path + "total_correlation.json", "wb") as f:
#        pick.dump(total_map, f)
#    with open(args.data_path + "emo_correlation.json", 'w') as f:
#        pick.dump(emo_res_map, f)
    precent_correlation.close() 
    topk_correlation.close() 

def sort_write_map(correlation_map, precent_file, topk_file, row_name, topk):
    map_items = correlation_map.items()
    values = correlation_map.values()
    print(f"values: {values}")
    correlation_num = float(sum(values))
    print(f"map length: {len(map_items)}, correlation_num: {correlation_num}")
    map_items = sorted(map_items, key=lambda x: x[1], reverse=True)
    print(f"map_items: {map_items}")
    precent_correlation = list(map(lambda x: (x[0], (x[1] / correlation_num) * 100), map_items))

    precent_result = [row_name + f"({int(correlation_num)})"]
    for i in range(20):
        precent = int((i) * 0.05 * len(map_items))
        print(f"i: {i}, precent: {precent}")
        top = sum([x[1] for x in precent_correlation[:precent]])
        precent_result.append(str(round(top, 2))) 
    precent_result.append(str(100.0))
    precent_file.write(",".join(precent_result))
    precent_file.write("\n")

    top_res = [row_name + f"({int(correlation_num)})"]
    top_correlation = list(map(lambda x: "-".join(x[0]) + f"({round(x[1], 2)})", precent_correlation[:topk]))
    top_res.extend(top_correlation)
    topk_file.write(",".join(top_res))
    topk_file.write("\n")
    
    top = sum([x[1] for x in precent_correlation[:30]])
    print(f"top: 30, {top}")
    top = sum([x[1] for x in precent_correlation[:50]])
    print(f"top: 50, {top}")
    top = sum([x[1] for x in precent_correlation[:100]])
    print(f"top: 100, {top}")
    top = sum([x[1] for x in precent_correlation[:200]])
    print(f"top: 200, {top}")
    percent_2 = int(0.2 * len(map_items))
    top = sum([x[1] for x in precent_correlation[:percent_2]])
    print(f"percent_2 top: {percent_2}, {top}")
    print(f"precent_correlation: {precent_correlation}")
#    return precent_correlation, correlation_num 

def update_dict(correlation_key, total_map, emo_res_map, emotion):
    if correlation_key is None:
        return
    if correlation_key not in total_map:
        total_map[correlation_key] = 0.0
    total_map[correlation_key] = total_map[correlation_key] + 1.0

    if correlation_key not in emo_res_map[emotion]:
        emo_res_map[emotion][correlation_key] = 0.0
    emo_res_map[emotion][correlation_key] = emo_res_map[emotion][correlation_key] + 1.0

def get_index(emo_index, dependencies):
    dep = dependencies[emo_index]
    relation, head_index, tail_index = dep[0], dep[1] - 1, dep[2] - 1
    return relation, head_index, tail_index

def find_by_emo_tail(emo_word, tokens, tags, dependencies):
    if emo_word not in tokens: 
        return None
    # if emotion word in tokens, find its index.
    emo_index = tokens.index(emo_word)
    emo_tag = tags[emo_index]
    relation, head_index, tail_index = get_index(emo_index, dependencies)
    #print(f"find_by_emo_tail: {emo_word}, {relation}, {head_index}, {tail_index}")
    if head_index > -1:
        head_tag = tags[head_index]
        head_word = tokens[head_index]
    else:
        head_tag = "ROOT"
        head_word = "ROOT"
    return ((head_tag, relation, emo_tag, "b"), (head_word, emo_word))

def find_by_emo_head(emo_word, tokens, tags, dependencies, predicted_heads):
    if emo_word not in tokens: 
        return None
    # if emotion word in tokens, find its index.
    emo_index = tokens.index(emo_word)
    emo_tag = tags[emo_index]

    tmp_emo_index = emo_index + 1
    # emo_word is not a head in context.
    if tmp_emo_index not in predicted_heads:
        return None
    emo_head_index = predicted_heads.index(tmp_emo_index)
    #print(f"emo_index: {emo_index}, tmp_emo_index: {tmp_emo_index}, emo_head_index: {emo_head_index}")

    relation, head_index, tail_index = get_index(emo_head_index, dependencies)
    #print(f"find_by_emo_head: {emo_word}, {relation}, {head_index}, {tail_index}")
    tail_tag = tags[tail_index]
    tail_word = tokens[tail_index]
    return ((emo_tag, relation, tail_tag, "f"), (emo_word, tail_word))
            
def main():
    statistics_correlations()
    # Get dependency annotation

if __name__ == "__main__":
    main()
