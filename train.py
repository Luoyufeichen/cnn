import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from globalvar import gloVar
import DP
import node
from torch.autograd import Variable
import numpy as np

def same_num(lg, target):
    # print(lg)
    sum = 0
    for i in range(len(lg)):

        if(lg[i] == target[i]):
            sum+=1
    return sum



def word_num(str, vocab): #根据词典将单词转成数字
    for i in range(len(vocab)):
        if vocab[i] == str.lower():
            return i
    return 0
def train(nd, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()


    train_data = nd.get_train_data()
    train_lable = nd.get_train_lable()
    label_voc = nd.get_label_voc()
    train_data_voc = nd.get_train_data_voc()


    for epoch in range(1, args.epochs+1):
        train_data_sentence_num = nd.get_train_data_sentence_num()
        for i in range(0, train_data_sentence_num, args.batch_size):
                feature = []
                target_1 = []
                if i+args.batch_size > train_data_sentence_num:
                    cc = train_data_sentence_num
                else:
                    cc = i + args.batch_size

                wc_max = 0
                for jj in range(i, cc):
                    wc_max = wc_max if wc_max > len(train_data[jj]) else len(train_data[jj])

                for j in range(i, cc):
                    target_1.append(word_num(train_lable[j], label_voc))
                    # target_1.append(0)
                    new_l1 = []
                    for k in range(len(train_data[j])):
                        new_l1.append(word_num(train_data[j][k], train_data_voc))
                    for kk in range(0, wc_max-len(new_l1)):
                        new_l1.append(1)
                    feature.append(new_l1)
                feature , target= Variable(torch.LongTensor(feature)), Variable(torch.LongTensor(target_1))

                optimizer.zero_grad()
                logit = model(feature)

                loss = F.cross_entropy(logit, target)
                loss.backward()
                optimizer.step()

                steps += 1
                if steps % args.log_interval == 0:
                    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                    # print(type((lg == target).astype(np.int32)))
                    accuracy = 100.0 * corrects / args.batch_size
                    sys.stdout.write(
                        '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                         loss.data[0],
                                                                         accuracy,
                                                                         corrects,
                                                                         args.batch_size))
                if steps % args.test_interval == 0:
                     dev_acc = eval(nd, model, args)
                     if dev_acc > best_acc:
                         best_acc = dev_acc
                         last_step = steps
                         if args.save_best:
                             save(model, args.save_dir, 'best', steps)
                     else :
                         if steps - last_step >= args.early_stop:
                             print('early stop by {} steps.'.format(args.early_stop))

def eval(nd, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    label_voc = nd.get_label_voc()
    train_data_voc = nd.get_train_data_voc()
    dev_data = nd.get_dev_data()
    dev_lable = nd.get_dev_lable()
    dev_sentence_num = nd.get_dev_sentence_num()

    for i in range(0, dev_sentence_num, args.batch_size):
        feature = []

        target_1 = []
        if i+args.batch_size > dev_sentence_num:
            cc = dev_sentence_num
        else:
            cc = i + args.batch_size

        wc_max = 0
        for jj in range(i, cc):
            wc_max = wc_max if wc_max > len(dev_data[jj]) else len(dev_data[jj])

        for j in range(i, cc):

            target_1.append(word_num(dev_lable[j], label_voc))
            new_l1 = []
            for k in range(len(dev_data[j])):
                new_l1.append(word_num(dev_data[j][k], train_data_voc))
            for kk in range(0, wc_max - len(new_l1)):
                new_l1.append(1)
            feature.append(new_l1)

        feature, target = Variable(torch.LongTensor(feature)), Variable(torch.LongTensor(target_1))
        # print(feature)

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(dev_data)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy
def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.txt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


