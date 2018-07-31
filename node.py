# -*- coding: utf-8 -*-
# @Time    : 18-7-28 上午9:20
# @Author  : WangZhen
# @FileName: node.py
# @Software: PyCharm Community Edition
class Node:
    def _init(self):
        train_data = []
        train_data_voc = []
        train_data_sentence_num = 0
        label_voc = []
        train_lable = []
        wc_max = 0
        dev_data = []
        dev_lable = []
        wc_max_dev = 0
        dev_sentence_num = 0
        word_list = []

    def get_train_data(self):
        return self.train_data
    def set_train_data(self, train_data):
        self.train_data = train_data

    def get_train_data_voc(self):
        return self.train_data_voc
    def set_train_data_voc(self, train_data_voc):
        self.train_data_voc = train_data_voc

    def get_train_data_sentence_num(self):
        return self.train_data_sentence_num
    def set_train_data_sentence_num(self, train_data_sentence_num):
        self.train_data_sentence_num = train_data_sentence_num

    def get_label_voc(self):
        return self.label_voc
    def set_label_voc(self, label_voc):
        self.label_voc = label_voc

    def get_train_lable(self):
        return self.train_lable
    def set_train_lable(self, train_lable):
        self.train_lable = train_lable



    def get_dev_data(self):
        return self.dev_data
    def set_dev_data(self, dev_data):
        self.dev_data = dev_data

    def get_dev_lable(self):
        return self.dev_lable
    def set_dev_lable(self, dev_lable):
        self.dev_lable = dev_lable



    def get_dev_sentence_num(self):
        return self.dev_sentence_num
    def set_dev_sentence_num(self, dev_sentence_num):
        self.dev_sentence_num = dev_sentence_num

    def get_word_list(self):
        return self.word_list
    def set_word_list(self, word_list):
        self.word_list = word_list