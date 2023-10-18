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

from allennlp.predictors.predictor import Predictor
from lxml import etree
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm
import pickle

from src.utils.data.loader import load_dataset


MODELS_DIR = './models'
model_path = os.path.join(
    MODELS_DIR, "biaffine-dependency-parser-ptb-2020.04.06.tar.gz")

def convert(n):
    return str(datetime.timedelta(seconds = n)) 

def wrapper_calc_time(print_log=True):
    """ 
    :param print_log: 
    :return:
    """

    def wrapper(func):
        def inner_wrapper(*args, **kwargs):
            start_time = time.time()
            func_re = func(*args, **kwargs)
            run_time = time.time() - start_time
            #re_time = f'{func.__name__}耗时：{int(tem_time * 1000)}ms'
            converted_time = convert(run_time)
            if print_log:
                print(f"{func.__name__} time:", run_time, converted_time)
            return func_re

        return inner_wrapper

    return wrapper


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--model_path', type=str, default=model_path,
                        help='Path to biaffine dependency parser.')
    parser.add_argument('--data_path', type=str, default='./data/ED/',
                        help='Directory of where ED data held.')
    parser.add_argument('--save_name', type=str, default='dep_tree.p',
                        help='Directory of where ED data held.')
    parser.add_argument('--data_type', type=str, default='context',
                        help='Directory of where ED data held.')
    parser.add_argument('--device', type=int, default=7)
    return parser.parse_args()


sentiment_map = {0: 'neutral', 1: 'positive', -1: 'negative'}


def read_file(file_name):
    '''
    Read twitter data and extract text and store.
    return sentences of [sentence, aspect_sentiment, from_to]
    '''
    with open(file_name, 'r') as f:
        data = f.readlines()
        data = [d.strip('\n') for d in data]
    # list of dict {text, aspect, sentiment}
    sentences = []
    idx = 0
    while idx < len(data):
        text = data[idx]
        idx += 1
        aspect = data[idx]
        idx += 1
        sentiment = data[idx]
        idx += 1
        sentence = get_sentence(text, aspect, sentiment)
        sentences.append(sentence)
    print(file_name, len(sentences))
#    with open(file_name.replace('.raw', '.txt'), 'w') as f:
#        for sentence in sentences:
#            f.write(sentence['sentence'] + '\n')

    return sentences


def get_sentence(text, aspect, sentiment):
    sentence = dict()
    sentence['sentence'] = text.replace('$T$', aspect)
    sentence['aspect_sentiment'] = [[aspect, sentiment_map[int(sentiment)]]]
    frm = text.split().index('$T$')
    to = frm + len(aspect.split())
    sentence['from_to'] = [[frm, to]]
    return sentence


def text2docs(file_path, predictor):
    '''
    Annotate the sentences from extracted txt file using AllenNLP's predictor.
    '''
    sentences = sentences[:1]
    docs = []
    print('Predicting dependency information...')
    for i in tqdm(range(len(sentences))):
        docs.append(predictor.predict(sentence=sentences[i]))

    return docs


def dependencies2format(doc):  # doc.sentences[i]
    '''
    Format annotation: sentence of keys
                                - tokens
                                - tags
                                - predicted_dependencies
                                - predicted_heads
                                - dependencies
    '''
    sentence = {}
    sentence['tokens'] = doc['words']
    sentence['tags'] = doc['pos']
    # sentence['energy'] = doc['energy']
    predicted_dependencies = doc['predicted_dependencies']
    predicted_heads = doc['predicted_heads']
    sentence['predicted_dependencies'] = doc['predicted_dependencies']
    sentence['predicted_heads'] = doc['predicted_heads']
    sentence['dependencies'] = []
    for idx, item in enumerate(predicted_dependencies):
        dep_tag = item
        frm = predicted_heads[idx]
        to = idx + 1
        sentence['dependencies'].append([dep_tag, frm, to])

    return sentence

def get_dep(sentence, predictor, tag_set, dep_set):
    res_sen = []
    #print(f"sentence:", sentence)
    for utterence in sentence: 
        u_length = len(utterence)
        utterence = " ".join(utterence)
        #print(f"utterence:", utterence)
        doc = predictor.predict(sentence=utterence)
        formated_doc = dependencies2format(doc)
        tag_set.update(formated_doc["tags"])

        #tags = formated_doc["tags"]
        #tokens = formated_doc["tokens"]
        #predicted_dependencies = formated_doc["predicted_dependencies"]
        #print(f"tags length: {len(tags)}, tokens length: {len(tokens)}, utterence: {u_length}, predicted_dependencies lenghth: {(len(predicted_dependencies))}")
        #print("====================================")
        dep_set.update(formated_doc["predicted_dependencies"])
        res_sen.append(formated_doc)
    return res_sen

@wrapper_calc_time(print_log=True)
def get_dependencies(data, predictor, tag_set, dep_set, data_type):
    context = data["context"]#[:100]
    target = data["target"]#[:100]
    situation = data["situation"]#[:100]
    res = {}
    res["context"] = []

    # context
    if data_type == "context":
        print("Process context")
        for sentence in context:
            res_sen = get_dep(sentence, predictor, tag_set, dep_set)
            res["context"].append(res_sen)
    elif data_type == "target":
        print("Process target")
        res["target"] = get_dep(target, predictor, tag_set, dep_set)
    elif data_type == "situation":
        print("Process situation")
        res["situation"] = get_dep(situation, predictor, tag_set, dep_set)
    return res

def syntaxInfo2json(sentences, sentences_with_dep, file_name):
    json_data = []
    # mismatch_counter = 0
    for idx, sentence in enumerate(sentences):
        sentence['tokens'] = sentences_with_dep[idx]['tokens']
        sentence['tags'] = sentences_with_dep[idx]['tags']
        sentence['predicted_dependencies'] = sentences_with_dep[idx]['predicted_dependencies']
        sentence['dependencies'] = sentences_with_dep[idx]['dependencies']
        sentence['predicted_heads'] = sentences_with_dep[idx]['predicted_heads']
        # sentence['energy'] = sentences_with_dep[idx]['energy']
        json_data.append(sentence)
    
#    with open(file_name.replace('.txt', '_biaffine.json'), 'w') as f:
#        json.dump(json_data, f)
#    print('done', len(json_data))


def main():
    args = parse_args()

    print("Loading tree model...")
    predictor = Predictor.from_path(args.model_path, cuda_device=args.device)
    #, dataset_reader_to_load="UD"

    print("Loading dataset...")
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    process_data = [pairs_tra, pairs_val, pairs_tst]

    # Get dependency annotation
    result = []
    tag_set = set()
    dep_set = set()
    data_type = args.data_type # context, target, situation
    for i, data in enumerate(process_data):
        print(f"get_dependencies for index: {i}")
        data_with_dep = get_dependencies(data, predictor, tag_set, dep_set, data_type)
        result.append(data_with_dep)

    result.append(list(tag_set))
    result.append(list(dep_set))
    print("Save data...")
    with open(args.data_path + data_type + "_"  + args.save_name, "wb") as f:
        pickle.dump(result, f)
#    with open(args.data_path + args.save_name, 'w') as f:
#        json.dump(result, f)


if __name__ == "__main__":
    main()
