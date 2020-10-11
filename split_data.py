from tqdm import tqdm
import os
import datetime,time
import json
from collections import Counter
import numpy as np
import pickle
import random
import logging
import multiprocessing
import yaml
import jieba,re
from matplotlib import pyplot as plt



Weibo_file="./data/Weibo.txt"
Weiboeids_file="./data/Weiboeids"
all_post_file="./Weibo/"
stop_word_file="./stopword.txt"
if __name__=='__main__':
    #将all_event_ids随机打乱划分训练集和测试集，并且存储起来
    all_event_ids = json.load(open(Weiboeids_file, 'r'))['train']#存储的是所有事件的event_id
    print(len(all_event_ids),type(all_event_ids))
    #长度为4664，代表有4664个谣言事件
    #根据打乱的data_index索引来随机的打乱各个事件
    data_index=list(range(len(all_event_ids)))
    print(data_index[-100:])
    r=random.random
    random.seed(4000)
    random.shuffle(data_index,random=r)
    print(data_index[-100:])

    print("-"*100)
    print(all_event_ids[:50])
    shuffled_event_ids=np.array(all_event_ids)[data_index].tolist()
    print(shuffled_event_ids[:50])

    train_data_nums=int(len(all_event_ids)*0.8)
    train_data_f=open("./train_event_ids.json","w")
    train_data={"train":shuffled_event_ids[:train_data_nums]}
    #余下20%作为测试
    test_data_f=open("./test_event_ids.json","w")
    test_data={"test":shuffled_event_ids[train_data_nums:]}

    json.dump(train_data,train_data_f,ensure_ascii=False)
    json.dump(test_data,test_data_f,ensure_ascii=False)

    train_data_f.close()
    test_data_f.close()




