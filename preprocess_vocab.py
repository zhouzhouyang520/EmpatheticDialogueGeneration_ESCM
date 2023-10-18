import datetime
import time

from src.utils.data.loader import *

def convert(n):
    return str(datetime.timedelta(seconds = n)) 

def wrapper_calc_time(print_log=True):
    """ 
    计算func执行时间
    :param print_log: 是否打印日志
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

@wrapper_calc_time(print_log=True)
def propocess_vocab():
    print("Loading data...")
    dep_tree_tra, dep_tree_val, dep_tree_tst, tag_set, dep_set = load_dep_tree()
    vocab = Lang(
        {
            config.UNK_idx: "UNK",
            config.PAD_idx: "PAD",
            config.EOS_idx: "EOS",
            config.SOS_idx: "SOS",
            config.USR_idx: "USR",
            config.SYS_idx: "SYS",
            config.CLS_idx: "CLS",
        })
    tree_list = [dep_tree_tra, dep_tree_val, dep_tree_tst]
    #tree_list = [dep_tree_tra, ]
    print("Propocess data...")
    tmp_set = set()
    for dep_tree in tree_list: # [dep_tree_tra, dep_tree_val, dep_tree_tst]
        for key in dep_tree:
            data_list = dep_tree[key]
            for data in data_list:#[:1]:
                if key in ["situation", "target"]:
                    #print(f"key: {key}, item: {data}")
                    item = data["tokens"]
                
                if key == "context":
                    #print(f"key: {key}, item: {data}")
                    item = []
                    for d in data:
                        item.extend(d["tokens"])
                
                tmp_set.update(item)
                vocab.index_words(item)
    print("vocab.index2word", vocab.index2word)
    print(f"vocab length: {vocab.n_words}, check_length: {len(tmp_set)}")
    print("Propocess data end, and save data...")
    with open("data/ED/vocab.p", "wb") as f:
        pickle.dump(vocab, f)

if __name__ == "__main__":
    propocess_vocab()
