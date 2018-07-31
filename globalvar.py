#!/usr/bin/python
# -*- coding: utf-8 -*-

class gloVar():
    state = False
    train_data_word = []
    dev_data_word = []

    def train_set_data(self, data):
        self.train_data_word = data

    def dev_set_data(self, data):
        self.dev_data_word = data

    def train_get_data(self):
        return self.train_data_word
    def dev_get_data(self):
        return self.dev_data_word



