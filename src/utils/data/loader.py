import os
import nltk
import json
import pickle
import logging
import numpy as np
from tqdm.auto import tqdm
from src.utils import config
import torch.utils.data as data
from src.utils.common import save_config
from nltk.corpus import wordnet, stopwords
from src.utils.constants import DATA_FILES
from src.utils.constants import EMO_MAP as emo_map
from src.utils.constants import WORD_PAIRS as word_pairs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
emotion_lexicon = json.load(open("data/NRCDict.json"))[0]
stop_words = stopwords.words("english")

import torch
#torch.set_printoptions(threshold=np.inf)
#np.set_printoptions(threshold=np.inf)

class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def process_sent(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence


def get_commonsense(comet, item, data_dict):
    cs_list = []
    input_event = " ".join(item)
    for rel in relations:
        cs_res = comet.generate(input_event, rel)
        cs_res = [process_sent(item) for item in cs_res]
        cs_list.append(cs_res)

    data_dict["utt_cs"].append(cs_list)


def encode_ctx(vocab, items, data_dict, comet):
    for ctx in tqdm(items):
        ctx_list = []
        e_list = []
        for i, c in enumerate(ctx):
            item = process_sent(c)
            ctx_list.append(item)
            vocab.index_words(item)
            ws_pos = nltk.pos_tag(item)  # pos
            for w in ws_pos:
                w_p = get_wordnet_pos(w[1])
                if w[0] not in stop_words and (
                    w_p == wordnet.ADJ or w[0] in emotion_lexicon
                ):
                    e_list.append(w[0])
            if i == len(ctx) - 1:
                get_commonsense(comet, item, data_dict)

        data_dict["context"].append(ctx_list)
        data_dict["emotion_context"].append(e_list)


def encode(vocab, files):
    from src.utils.comet import Comet

    data_dict = {
        "context": [],
        "target": [],
        "emotion": [],
        "emotion_context": [],
        "utt_cs": [],
    }
    comet = Comet("data/Comet", config.device)

    for i, k in enumerate(data_dict.keys()):
        items = files[i]
        if k == "context":
            encode_ctx(vocab, items, data_dict, comet)
        elif k == "emotion":
            data_dict[k] = items
        else:
            for item in tqdm(items):
                item = process_sent(item)
                data_dict[k].append(item)
                vocab.index_words(item)
        if i == 3:
            break
    assert (
        len(data_dict["context"])
        == len(data_dict["target"])
        == len(data_dict["emotion"])
        == len(data_dict["emotion_context"])
        == len(data_dict["utt_cs"])
    )

    return data_dict


def read_files(vocab):
    files = DATA_FILES(config.data_dir)
    train_files = [np.load(f, allow_pickle=True) for f in files["train"]]
    dev_files = [np.load(f, allow_pickle=True) for f in files["dev"]]
    test_files = [np.load(f, allow_pickle=True) for f in files["test"]]

    data_train = encode(vocab, train_files)
    data_dev = encode(vocab, dev_files)
    data_test = encode(vocab, test_files)

    return data_train, data_dev, data_test, vocab

def load_dataset():
    data_dir = config.data_dir
    cache_file = f"{data_dir}/dataset_preproc.p"
    if os.path.exists(cache_file):
        print("LOADING empathetic_dialogue")
        with open(cache_file, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab = read_files(
            vocab=Lang(
                {
                    config.UNK_idx: "UNK",
                    config.PAD_idx: "PAD",
                    config.EOS_idx: "EOS",
                    config.SOS_idx: "SOS",
                    config.USR_idx: "USR",
                    config.SYS_idx: "SYS",
                    config.CLS_idx: "CLS",
                }
            )
        )
        with open(cache_file, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")
#    for i in range(20):
#        print("[emotion]:", data_tra["emotion"][i])
#        print("[context]:", [" ".join(u) for u in data_tra["context"][i]])
#        print("[emotion_context]:", data_tra["emotion_context"][i])
#        print("[target]:", " ".join(data_tra["target"][i]))
#        print(" ")
    return data_tra, data_val, data_tst, vocab


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab, dep_tree, dep_tree_vocab, tree_dict):

        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        self.emo_map = emo_map
        self.analyzer = SentimentIntensityAnalyzer()
        self.dep_tree, self.dep_tree_vocab = dep_tree, dep_tree_vocab
        self.tree_dict = tree_dict
        self.tree_weight_num = 5
        

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["emotion_context"] = self.data["emotion_context"][index]
    
        item["context_text"], item["context_parents"] = self.get_context_with_parents(self.dep_tree["context"][index])
        item["target_text"] = self.dep_tree["target"][index]["tokens"]
        item["context_emotion_scores"] = self.analyzer.polarity_scores(
            " ".join(self.data["context"][index][0])
        )

        item["context"], item["context_mask"], item["context_ext"], item["context_tree"], item["context_tree_relation"], item["concetxt_tree_weight"] = self.preprocess(item["context_text"])
        item["target"] = self.preprocess(item["target_text"], anw=True)

        item["context_tags"] = self.preprocess(self.dep_tree["context"][index], tags=True)
        item["relation_matrix"] = self.preprocess(self.dep_tree["context"][index], tree=True)
        item["tags_text"] = self.dep_tree["context"][index]

        item["emotion"], item["emotion_label"] = self.preprocess_emo(
            item["emotion_text"], self.emo_map
        )
        (
            item["emotion_context"],
            item["emotion_context_mask"],
            _,
            _,
            _,
            _,
        ) = self.preprocess(item["emotion_context"])

        item["cs_text"] = self.data["utt_cs"][index]
        item["x_intent_txt"] = item["cs_text"][0]
        item["x_need_txt"] = item["cs_text"][1]
        item["x_want_txt"] = item["cs_text"][2]
        item["x_effect_txt"] = item["cs_text"][3]
        item["x_react_txt"] = item["cs_text"][4]

        item["x_intent"] = self.preprocess(item["x_intent_txt"], cs=True)
        item["x_need"] = self.preprocess(item["x_need_txt"], cs=True)
        item["x_want"] = self.preprocess(item["x_want_txt"], cs=True)
        item["x_effect"] = self.preprocess(item["x_effect_txt"], cs=True)
        item["x_react"] = self.preprocess(item["x_react_txt"], cs="react")

        return item

    def get_context_with_parents(self, dep_tree_data):
        context = []
        dep_tree_heads = [0]
        for content in dep_tree_data:
            context.append(content["tokens"])
            dep_tree_heads.extend(content["predicted_heads"]) 
        return context, torch.LongTensor(dep_tree_heads)

    def process_oov(self, sentence, ids, oovs):
        for w in sentence:
            if w in self.vocab.word2index:
                i = self.vocab.word2index[w]
                ids.append(i)
            else:
                if w not in oovs:
                    oovs.append(w)
                oov_num = oovs.index(w)
                ids.append(len(self.vocab.word2index) + oov_num)

    def process_context_oov(self, context):  #
        ids = []
        oovs = []
        for si, sentence in enumerate(context):
            self.process_oov(sentence, ids, oovs)
        return ids, oovs

    def preprocess(self, arr, anw=False, cs=None, emo=False, tags=False, tree=False):
        """Converts words to ids."""
        if anw:
            sequence = [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in arr
            ] + [config.EOS_idx]

            return torch.LongTensor(sequence)
        elif cs:
            sequence = [config.CLS_idx] if cs != "react" else []
            for sent in arr:
                sequence += [
                    self.vocab.word2index[word]
                    for word in sent
                    if word in self.vocab.word2index and word not in ["to", "none"]
                ]

            return torch.LongTensor(sequence)
        elif emo:
            x_emo = [config.CLS_idx]
            x_emo_mask = [config.CLS_idx]
            for i, ew in enumerate(arr):
                x_emo += [
                    self.vocab.word2index[ew]
                    if ew in self.vocab.word2index
                    else config.UNK_idx
                ]
                x_emo_mask += [self.vocab.word2index["CLS"]]

            assert len(x_emo) == len(x_emo_mask)
            return torch.LongTensor(x_emo), torch.LongTensor(x_emo_mask)
        elif tags:
            tag_list = []
            for data in arr:
                tag_list.extend(data["tags"])
            tag_list = ["CLS"] + tag_list
            tag_index = list(map(lambda x: self.dep_tree_vocab.word2index[x], tag_list))
            return torch.IntTensor(tag_index)
        elif tree:
            dep_list = []
            tmp_dep_list = []
            base_index = 0
            
            tmp_list = []
            tmp_tokens = []
            for data in arr:
                dep = data["dependencies"]
                tmp_list.extend(dep)
                tmp_tokens.append(data["tokens"])
            for data in arr:
                dep = data["dependencies"]
                for i, rel in enumerate(dep):
                    rel_type, head, tail = rel[0], rel[1], rel[2]
                    if head == 0: # if head is root, point to it self.
                        head = tail
                    head = base_index + head # + 1
                    tail = base_index + tail
                    dep_list.append([rel_type, head, tail])
                base_index += len(dep)
                tmp_dep_list.append(data["dependencies"])

            length = len(dep_list) + 1 # CLS + arr
            temp_adjacency_matrix = np.zeros([length, length])
            relation_matrix = np.full([length, length], fill_value=self.dep_tree_vocab.word2index["PAD"])
            # Set CLS relation_matrix
            relation_matrix[0, :] = self.dep_tree_vocab.word2index["CREL"]
            relation_matrix[:, 0] = self.dep_tree_vocab.word2index["CREL"]
            for rel in dep_list:
                rel_type, head, tail = rel[0], rel[1], rel[2]
                relation_matrix[head, tail] = self.dep_tree_vocab.word2index[rel_type]
            return torch.LongTensor(relation_matrix)
        else:
            # Count the length of sentences.
            length = 1
            for i, sentence in enumerate(arr):
                length += len(sentence)
            
            tree_matrix = np.ones([length, config.tree_num], dtype=int)
            tree_relation_matrix = np.ones([length, config.tree_num], dtype=int)
            tree_weight_matrix = np.zeros([length, config.tree_num, self.tree_weight_num])
            x_dial = [config.CLS_idx]
            x_mask = [config.CLS_idx]
            x_ext, x_oovs = self.process_context_oov(arr)
            x_dial_ext = [config.CLS_idx] + x_ext
            j = 0
            for i, sentence in enumerate(arr):
                for word in sentence:
                    j += 1
                    if word in self.vocab.word2index:
                        index = self.vocab.word2index[word]
                    else:
                        index = config.UNK_idx
                    x_dial.append(index)
                    if word in self.tree_dict:
                        for k, tree in enumerate(self.tree_dict[word][:config.tree_num]):
                            if tree[0] in self.vocab.word2index:
                                tree_word = tree[0] 
                                tree_relation = tree[1]
                                tree_matrix[j, k] = self.vocab.word2index[tree_word]
                                tree_relation_matrix[j, k] = self.dep_tree_vocab.word2index[tree_relation]
                                tree_weight_matrix[j, k, :] = np.array(tree[2:])
                
                spk = (
                    self.vocab.word2index["USR"]
                    if i % 2 == 0
                    else self.vocab.word2index["SYS"]
                )
                x_mask += [spk for _ in range(len(sentence))]
            assert len(x_dial) == len(x_mask)

            return torch.LongTensor(x_dial), torch.LongTensor(x_mask), torch.LongTensor(x_dial_ext), torch.LongTensor(tree_matrix), torch.LongTensor(tree_relation_matrix), torch.FloatTensor(tree_weight_matrix)

    def preprocess_emo(self, emotion, emo_map):
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(
            len(sequences), max(lengths)
        ).long()  ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def merge_matrix(sequences, rel):
        length_list = []
        max_x = 0
        max_y = 0

        for seq in sequences:
            shape = seq.shape
            length_list.append(shape)
            if shape[0] > max_x:
                max_x = shape[0]
            if shape[1] > max_y:
                max_y = shape[1]

        if rel:
            padded_seqs = torch.ones(
                len(sequences), max_x, max_y
            ).long() 
        else:
            padded_seqs = torch.zeros(
                len(sequences), max_x, max_y
            ).long() 
        for i, seq in enumerate(sequences):
            shape = length_list[i]
            padded_seqs[i, :shape[0], :shape[1]] = seq
        return padded_seqs

    def merge_weight_matrix(sequences):
        length_list = []
        max_x = 0
        max_y = 0
        dim = 0 # 5

        for seq in sequences:
            shape = seq.shape
            dim = shape[2]
            length_list.append(shape)
            if shape[0] > max_x:
                max_x = shape[0]
            if shape[1] > max_y:
                max_y = shape[1]
        padded_seqs = torch.zeros(len(sequences), max_x, max_y, dim)

        for i, seq in enumerate(sequences):
            shape = length_list[i]
            padded_seqs[i, :shape[0], :shape[1], :] = seq
        return padded_seqs

    data.sort(key=lambda x: len(x["context"]), reverse=True)  ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    input_batch, input_lengths = merge(item_info["context"])
    mask_input, mask_input_lengths = merge(item_info["context_mask"])
    emotion_batch, emotion_lengths = merge(item_info["emotion_context"])
    parents_batch, parents_lengths = merge(item_info["context_parents"])
    
    input_ext_batch, _ = merge(item_info["context_ext"])

    tag_batch, _ = merge(item_info["context_tags"])

    relation_matrix = merge_matrix(item_info["relation_matrix"], rel=True)
    
    context_tree = merge_matrix(item_info["context_tree"], rel=True)
    context_tree_relation = merge_matrix(item_info["context_tree_relation"], rel=True)
    context_tree_weight = merge_weight_matrix(item_info["concetxt_tree_weight"])

    ## Target
    target_batch, target_lengths = merge(item_info["target"])

    input_batch = input_batch.to(config.device)
    mask_input = mask_input.to(config.device)
    target_batch = target_batch.to(config.device)
    input_ext_batch = input_ext_batch.to(config.device)
    d = {}
    d["input_batch"] = input_batch
    d["input_ext_batch"] = input_ext_batch
    d["input_lengths"] = torch.LongTensor(input_lengths)
    d["mask_input"] = mask_input
    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)
    d["emotion_context_batch"] = emotion_batch.to(config.device)
    d["enc_parents"] = parents_batch.int().to(config.device)

    d["tag_batch"] = tag_batch.to(config.device)
    d["relation_matrix"] = relation_matrix.to(config.device)

    # Concept
    d["context_tree"] = context_tree.to(config.device) 
    d["context_tree_relation"] = context_tree_relation.to(config.device) 
    d["context_tree_weight"] = context_tree_weight.to(config.device) 
    

    ##program
    d["target_program"] = item_info["emotion"]
    d["program_label"] = item_info["emotion_label"]

    ##text
    d["input_txt"] = item_info["context_text"]
    d["target_txt"] = item_info["target_text"]
    d["program_txt"] = item_info["emotion_text"]

    d["context_emotion_scores"] = item_info["context_emotion_scores"]

    relations = ["x_intent", "x_need", "x_want", "x_effect", "x_react"]
    for r in relations:
        pad_batch, _ = merge(item_info[r])
        pad_batch = pad_batch.to(config.device)
        d[r] = pad_batch
        d[f"{r}_txt"] = item_info[f"{r}_txt"]

    return d

def load_idf(load_path="data/ED/updated_vocab_idf.json"):
    with open(load_path, 'r') as f:
        print("LOADING vocabulary idf")
        idf_json = json.load(f)
    max_idf = 0.
    mean_idf = 0.0 
    min_idf = 99.0
    for key in idf_json:
        idf = idf_json[key]
        if max_idf < idf:
            max_idf = idf 
        if min_idf > idf:
            min_idf = idf 
        mean_idf += idf 
    print(f"Max idf: {max_idf}, Mean idf: {mean_idf / len(idf_json)}, Min idf: {min_idf}")
    return idf_json 

def load_vad(vad_path="data/ED/VAD.json"):
    VAD = json.load(open(vad_path, "r", encoding="utf-8"))  # NRC_VAD
    return VAD 

def load_tree_dict(tree_path="data/ED/ConceptNet_VAD_dict.json"):
    tree_dict = json.load(open(tree_path, "r", encoding="utf-8"))
    return tree_dict 

def build_tag_dep_vocab(tag_set, dep_set):
    print("Building tags and depdency tree vocab...")
    # NREL: no relation, CREL: the relations between CLS and words.
    vocab = Lang({0: "UNK", 1: "PAD", 2: "NREL", 3: "CLS", 4: "CREL"}) 
    vocab.index_words(tag_set)
    vocab.index_words(dep_set)
    return vocab

def load_dep_tree(load_path="data/ED/dep_tree.p"):
    with open(load_path, "rb") as f:
        [data_tra, data_val, data_tst, tag_set, dep_set] = pickle.load(f)
    return data_tra, data_val, data_tst, tag_set, dep_set

def load_vocab(load_path="data/ED/vocab.p"):
    with open(load_path, "rb") as f:
         vocab = pickle.load(f)
    return vocab

def prepare_data_seq(batch_size=32):
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    dep_tree_tra, dep_tree_val, dep_tree_tst, tag_set, dep_set = load_dep_tree()
    tree_dict = load_tree_dict()
    # update vocab
    vocab, dep_tree_vocab = load_vocab()
    print(f"dep_tree_vocab: {dep_tree_vocab.word2index}")

    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab, dep_tree_tra, dep_tree_vocab, tree_dict)
    data_loader_tra = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dataset_valid = Dataset(pairs_val, vocab, dep_tree_val, dep_tree_vocab, tree_dict)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dataset_test = Dataset(pairs_tst, vocab, dep_tree_tst, dep_tree_vocab, tree_dict)
    data_loader_tst = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    save_config()
    return (
        data_loader_tra,
        data_loader_val,
        data_loader_tst,
        vocab,
        dep_tree_vocab, 
        len(dataset_train.emo_map),
    )

