#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import cnn_model
import train
import DP
import node
parser = argparse.ArgumentParser(description='CNN text classification')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=50, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=300, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='1,3,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
parser.add_argument('-init-weight-decay', type = float, default=1e-8, help='l2 the parameters of weight decay')
parser.add_argument('-batch-normalizations', type = bool, default=True, help='batch_normalizations')
parser.add_argument('-bath-norm-momentum', type = float, default=0.1, help='bath_norm_momentum')
parser.add_argument('-batch-norm-affine', type = bool, default=True, help='batch_norm_affine')
parser.add_argument('-wide-conv', type = bool, default=True, help='wide_conv')
parser.add_argument('-seed-num', type = int , default=233, help='in order to chang the par of k_dim')
parser.add_argument('-out-word-v', type = bool, default=True, help='whether to add out-word-v')
parser.add_argument('-word-embed', type  = list, default=[], help='worc-v')

# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

def init():
    dict = {}
    with open('converted_word_Twitter.txt', encoding="utf-8") as f:
        for line in f:
            sentence = line.strip().split(' ')
            dict[sentence[0]] = sentence[1:]
    return dict

def word_v(str, dict):

    if str in dict:
        numbers = dict[str]
        numbers = [float(x) for x in numbers]
        return numbers
    sentence = []
    for i in range(300):
        sentence.append(0.0)
    return sentence

def add_word_v(train_data_voc_):
    dict = init()
    word_list = []
    for i in range(len(train_data_voc_)):
        lt = word_v(train_data_voc_[i], dict)
        word_list.append(lt)
    return word_list

def loaddata(args):
    nd = node.Node()
    nd.__init__()
    dp = DP.DataPrecess()
    train_data_voc, train_data_sentence_num = dp.buildvocab(path="raw.clean.train")  # type:list
    label_voc = dp.buildvocab_label(path="classfication")

    train_data, train_lable = dp.readdata_d(path="raw.clean.train", shuffle=args.shuffle)
    dev_data, dev_lable = dp.readdata_d(path="raw.clean.dev", shuffle=args.shuffle)
    unusr, dev_sentence_num = dp.readdata_v(path="raw.clean.dev")
    print('train_sentence:', train_data_sentence_num, 'dev_sentence:', dev_sentence_num)
    nd.set_dev_data(dev_data)
    nd.set_dev_lable(dev_lable)
    nd.set_dev_sentence_num(dev_sentence_num)

    nd.set_train_data(train_data)
    nd.set_train_data_sentence_num(train_data_sentence_num)
    nd.set_train_lable(train_lable)
    nd.set_train_data_voc(train_data_voc)

    nd.set_label_voc(label_voc)

    if args.out_word_v is True:
        args.word_embed = add_word_v(train_data_voc)

    for i in range(len(args.word_embed)):
        if len(args.word_embed[i]) != 300:
            print('less 300')

    args.embed_num = len(train_data_voc)
    args.class_num = len(label_voc)

    args.cuda = (not args.no_cuda) and torch.cuda.is_available();
    del args.no_cuda
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if attr == 'word_embed':
            continue
        print("\t{}={}".format(attr.upper(), value))

    return nd


nd = loaddata(args)

cnn = cnn_model.CNN_Text(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()
train.train(nd, cnn, args)
