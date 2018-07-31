import os
import random
import tarfile
import urllib
import re

#数据处理部分

class DataPrecess:
    def __init__(self):
        print("loading the data ......")

    def readdata_v(self, path = None):
        sentence =""
        i = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                sentence += ' '
                s, flag = line.strip().split('|||')
                s = self.clean_str(s)
                flag = flag.strip()
                if flag == '2':
                    continue
                sentence = sentence + s
                i = i+1
        text_inorder = self.wordcount(sentence)
        return text_inorder, i

    def readdata_d(self, path = None, shuffle = True):
        train_data = []
        train_lable = []
        if shuffle:
           os.system('shuf '+path+ ' -o ' + path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                s, flag = line.strip().split('|||')
                s = self.clean_str(s)
                flag = flag.strip()
                if flag == '0':
                    train_lable.append('negative')
                if flag == '1':
                    train_lable.append('negative')
                if flag == '2':
                    continue
                if flag == '3':
                    train_lable.append('positive')
                if flag == '4':
                    train_lable.append('positive')
                new_list = s.strip().split(' ')
                train_data.append(new_list)
        return  train_data, train_lable

    def wordcount(self, str):
        strl_ist = str.replace('\n', ' ').lower().split(' ')
        count_dict={}
        for str in strl_ist:
            if str in count_dict.keys():
                count_dict[str] = count_dict[str] + 1
            else:
                count_dict[str] = 1
        count_list = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        return count_list

    def buildvocab(self, path = None):
        vocab_list , sentence_num= self.readdata_v(path)

        for i in range(len(vocab_list)):
            if vocab_list[i][0] == '':
                del vocab_list[i]
                break
        #    print (i,":",vocab_dit[i][0])
        vl = []
        vl.append('<unk>')
        vl.append('<pad>')

        for i in range(len(vocab_list)):
            vl.append(vocab_list[i][0])
        return vl, sentence_num

    def buildvocab_label(self, path = None):
        vl = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip().split(' ')

        for i in range(len(s)):
            if(s[i] != ''):
                vl.append(s[i])

        return vl




    def clean_str(self,string):
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
        return string.strip()







