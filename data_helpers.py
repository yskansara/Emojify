import numpy as np
import re
import itertools
from collections import Counter
import sys
sys.path.insert(0, '../lstm')
from data_util import clean_emojis, generate_emoji_labels5
from util import emoji_dist, t_emoji_to_id


#define dummy args
class Object(object):
    pass

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
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
    return string.strip().lower()


def load_data_and_labels(top_emoji_file, data_file, top_emojis=None):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    data_raw = clean_emojis([line.split() for line in open(data_file, 'r')])
    if not top_emojis:
        TOP_TO_KEEP = 50
        emoji_dict = emoji_dist(data_raw) # grab the top 50 emojis
        top_emojis = zip(*emoji_dict.most_common(TOP_TO_KEEP))[0] #get top emojis with index
        f = open(top_emoji_file, 'w')
        f.write("\n".join(top_emojis))
        f.close()

    print "top_emojis"
    print top_emojis
    labeled = generate_emoji_labels5(data_raw, top_emojis)
    x_sent = []
    y = []

    for sent, emoji in labeled:
        x_sent.append(' '.join(sent))
        y.append(t_emoji_to_id(emoji[0]))
    return [x_sent, np.array(y)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        print "Epoch ", epoch
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


#############################
# Copied over from RNN work #
#############################


