import numpy as np
import cPickle
from collections import defaultdict
import sys
import re
import os
import pandas as pd
import random


def adjust(num_list):
    new_list = list()
    for num in num_list:
        if num == 0:
            new_list.append(num)
        elif num < 70:
            new_list.append(70)
        elif num > 117:
            new_list.append(117)
        else:
            new_list.append(num)
    return new_list


def build_data_pf(data_pkl, clean_string=True):
    revs = []
    vocab = defaultdict(float)
    row = 1
    dataset = cPickle.load(open(data_pkl))
    max_l, max_l2 = sent_max_length(dataset)
    for num, sent_info in dataset.items():
        sent_list = [str(token["word"]).lower() for token in sent_info["sent"]]
        orig_rev = " ".join(sent_list)
        words = set(sent_list)
        for word in words:
            vocab[word] += 1
        # generate PF string
        e1_id = sent_info["entity_pair"]["e1"]["id"]
        e2_id = sent_info["entity_pair"]["e2"]["id"]
        pf1_list = adjust([(token["id"] - e1_id + max_l)
                           for token in sent_info["sent"]])
        pf2_list = adjust([(token["id"] - e2_id + max_l)
                           for token in sent_info["sent"]])
        #pf1 = ' '.join([str(idx) for idx in pf1_list])
        #pf2 = ' '.join([str(idx) for idx in pf2_list])

        datum = {"y": int(sent_info["class"]),
                 "text": orig_rev,
                 "e1_to_e2": " ".join(sent_info['e1_to_e2']),
                 "num_words": len(orig_rev.split()),
                 "split": int(sent_info['split']),
                 # "split": int(num >= 8000),
                 "row": int(row),
                 "pf1": pf1_list,
                 "pf2": pf2_list,
                 "entity_word": [sent_list[e1_id], sent_list[e2_id]]}
        revs.append(datum)
        row += 1

    return revs, vocab, max_l, max_l2


def sent_max_length(dataset):
    #dataset = cPickle.load(open(data_pkl))
    length_list = list()
    length_e1_e2 = list()
    for key, sent_info in dataset.items():
        length_list.append(len(sent_info["sent"]))
        length_e1_e2.append(len(sent_info['e1_to_e2']))
        if len(sent_info["sent"]) > 70:
            print key
    # print list(set(length_list))[-100:]
    return max(length_list), max(length_e1_e2)


def get_idx_from_sent(sent, entity, word_idx_map, max_l=56, k=300, filter_h=2, flag=False):
    x = []
    iden = []
    mword = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
        iden.append(0)
        mword.append(1)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
            iden.append(0)
            mword.append(1)
    count = len(iden) + pad
    while len(iden) < count:
        iden.append(0)
        mword.append(1)

    while len(iden) < max_l + 2 * pad:
        iden.append(0 - 100000)
        mword.append(0)

    while len(x) < max_l + 2 * pad:
        x.append(0)
    if flag:
        x.append(word_idx_map[entity[0]])
        x.append(word_idx_map[entity[1]])
    return x, iden, mword


def get_idx_from_pf(pf_list, max_l=56, k=300, filter_h=2):
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        # x.append(pf_list[0]-1)
        x.append(0)
    x.extend(pf_list)
    count = len(x) + pad
    while len(x) < count:
        # x.append(pf_list[-1]+1)
        x.append(0)
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k))
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(
                    f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def clean_str(string, TREC=False):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def clean_str_sst(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

if __name__ == "__main__":
    w2v_file = "../GoogleNews-vectors-negative300.bin"  # sys.argv[1]
    data_pkl = '../SemEval_Data/data_2_label.pkl'  # sys.argv[2]
    data_dic = 'data_vector_luyao_2label'
    if not os.path.exists(data_dic):
        os.mkdir(data_dic)

    print "loading data...",
    revs, vocab, max_l, max_l2 = build_data_pf(data_pkl, clean_string=True)
    print revs[1]
    #max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "max sentence e1_to_e2 length: " + str(max_l2)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    # print type(W)
    # print W.shape
    if not os.path.exists(data_dic):
        os.mkdir(data_dic)
    np.savez(data_dic + '/embedding_weights.npz', embedding_weights=W)

    f_idx_map = open(data_dic + '/w2v_index', 'w')
    for item in word_idx_map:
        f_idx_map.write(item + '\t' + str(word_idx_map[item]) + '\n')

    train, train_e1e2_phrase, train_pf1, train_pf2, test, test_e1e2_phrase, test_pf1, test_pf2, train_y, test_y, train_vector, test_vector, train_mword, test_mword = [
    ], [], [], [], [], [], [], [], [], [], [], [], [], []
    e_id = []
    # random.shuffle(revs)
    for rev in revs:
        sent, iden, mword = get_idx_from_sent(
            rev["text"], rev["entity_word"], word_idx_map, max_l=max_l, k=300, filter_h=2, flag=True)
        e1e2_phrase, iden_2, mword_2 = get_idx_from_sent(
            rev["e1_to_e2"], rev["entity_word"], word_idx_map, max_l=max_l2, k=300, filter_h=5)

        pf1 = get_idx_from_pf(rev["pf1"], max_l=max_l, k=300, filter_h=2)
        pf2 = get_idx_from_pf(rev["pf2"], max_l=max_l, k=300, filter_h=2)
        # sent.append(rev["y"])
        if int(rev["split"]) == 1:
            #test.append([list(W[i]) for i in sent])
            test_e1e2_phrase.append(e1e2_phrase)
            test.append(sent)
            test_y.append(rev["y"])
            test_pf1.append(pf1)
            test_pf2.append(pf2)
            test_vector.append(iden)
            test_mword.append(mword)
            # test_e_id.append(rev["entity_id"])
        else:
            #train.append([list(W[i]) for i in sent])
            train_e1e2_phrase.append(e1e2_phrase)
            train.append(sent)
            train_y.append(rev["y"])
            train_pf1.append(pf1)
            train_pf2.append(pf2)
            train_vector.append(iden)
            train_mword.append(mword)
            # train_e_id.append(rev["entity_id"])

    np.set_printoptions(threshold='nan')

    train_e1e2_phrase = np.array(train_e1e2_phrase, dtype="int")
    test_e1e2_phrase = np.array(test_e1e2_phrase, dtype="int")

    train = np.array(train, dtype="int")
    test = np.array(test, dtype="int")
    train_y = np.array(train_y, dtype="int").reshape((8000, 1))
    test_y = np.array(test_y, dtype="int").reshape((2717, 1))
    train_pf1 = np.array(train_pf1, dtype="int")
    train_pf2 = np.array(train_pf2, dtype="int")
    test_pf1 = np.array(test_pf1, dtype="int")
    test_pf2 = np.array(test_pf2, dtype="int")

    train_all = np.concatenate((train, test), axis=0)
    train_y_all = np.concatenate((train_y, test_y), axis=0)
    train_pf1_all = np.concatenate((train_pf1, test_pf1), axis=0)
    train_pf2_all = np.concatenate((train_pf2, test_pf2), axis=0)
    #train_vector_ori = np.array(train_all[:, :105], dtype='bool')

    train_vector = np.array(train_vector, dtype='float32')
    train_mword = np.array(train_mword, dtype='float32')

    #test_vector_ori = np.array(test[:, :105], dtype='bool')
    test_vector = np.array(test_vector, dtype='float32')
    test_mword = np.array(test_mword, dtype='float32')

    train_vector_all = np.concatenate((train_vector, test_vector), axis=0)
    train_mword_all = np.concatenate((train_mword, test_mword), axis=0)
    train_e1e2_phrase_all = np.concatenate(
        (train_e1e2_phrase, test_e1e2_phrase), axis=0)

    print train_all[0]
    print 'inden', train_vector_all.shape
    print train_vector[0]
    print train_mword_all.shape
    print 'mword', train_mword_all[0]
    print train_y.shape
    print train_pf1.shape
    print train_pf1[0]
    print train_pf2[0]
    print train_e1e2_phrase.shape
    #train_shuffle = np.concatenate((train, train_y), axis=1)
    #test_shuffle = np.concatenate((test, test_y), axis=1)
    # np.random.shuffle(train_shuffle)

    # np.savez('./data_vector_new/'+'train_x.npz',train)
    # np.savez('./data_vector_new/'+'train_y.npz',train_y)
    np.random.seed(123)
    # np.random.shuffle(train)
    # np.random.shuffle(train_y)
    # np.random.shuffle(train_pf1)
    # np.random.shuffle(train_pf2)

    np.savez(data_dic + '/train_e1e2_phrase.npz', train_e1e2_phrase_all)
    np.savez(data_dic + '/test_e1e2_phrase.npz', test_e1e2_phrase)

    np.savez(data_dic + '/train_x.npz', train_all)
    np.savez(data_dic + '/train_y.npz', train_y_all)
    np.savez(data_dic + '/test_x.npz', test)
    np.savez(data_dic + '/test_y.npz', test_y)
    np.savez(data_dic + '/train_pf1.npz', train_pf1_all)
    np.savez(data_dic + '/train_pf2.npz', train_pf2_all)
    np.savez(data_dic + '/test_pf1.npz', test_pf1)
    np.savez(data_dic + '/test_pf2.npz', test_pf2)
    np.savez(data_dic + '/train_vector.npz', train_vector_all)
    np.savez(data_dic + '/test_vector.npz', test_vector)
    np.savez(data_dic + '/train_mword.npz', train_mword_all)
    np.savez(data_dic + '/test_mword.npz', test_mword)

    print "dataset created!"
