import re
import numpy as np


class Data(object):
    def __init__(self, json_data,
                 input_size=250):
        self.data = json_data
        self.length = input_size
        self.alphabet = ''
        self.alphabet_size = 0
        self.dict = {}  # Maps each character to an integer
        self.str_data = ''

    def preprocess_data(self):
        self.str_data = self.data
        self.str_data = re.sub('^\s*(.-)\s*$', '%1', self.str_data).replace('\\n', '\n')
        self.str_data = re.sub('\s+', ' ', self.str_data)
        self.str_data = self.str_data.lower()

    def set_alphabet(self, alphabet):
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}
        for idx, char in enumerate(self.alphabet):
            self.dict[char] = idx + 1

    def preprocess_type_text(self):
        self.str_data = self.str_data.replace("заробітна плата та аванси", "шаблон платежа")
        self.str_data = re.sub("\d+", "#", self.str_data)

    def get_data(self):
        batch_indices = []
        batch_indices.append(self.str_to_indexes(self.str_data))
        return np.asarray(batch_indices, dtype='int64')

    def str_to_indexes(self, s):
        """
        Convert a string to character indexes based on character dictionary.

        Args:
            s (str): String to be converted to indexes
        Returns:
            str2idx (np.ndarray): Indexes of characters in s
        """
        max_length = min(len(s), self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        for i in range(1, max_length):
            c = s[i]
            if c in self.dict:
                str2idx[i - 1] = self.dict[c]
        return str2idx


def from_percent(pred, treshold, status):
    temp_status = status
    result = 1
    if pred[0] < treshold:
        result = None
        temp_status = 0
    return result, temp_status


def from_categorical(pred, treshold, bias, status):
    temp_status = status
    max_ind = np.argmax(pred, axis=-1)
    temp_val = pred[0, max_ind[0]]
    if temp_val < treshold or max_ind[0] == 0:
        result = None
        temp_status = 0
    else:
        result = int(max_ind[0]) + bias
    return result, temp_status
