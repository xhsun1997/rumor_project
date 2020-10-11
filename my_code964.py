from tqdm import tqdm
import os
import datetime,time
import json
from collections import Counter
import numpy as np
import tensorflow as tf
import pickle
import random
from tensorflow.keras import backend
from tensorflow.keras.layers import BatchNormalization,Layer, Input,Conv2D,MaxPool2D,concatenate,Flatten,Dense,Dropout,Embedding,Reshape,LSTM,GRU
from tensorflow.keras import Sequential,optimizers,losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
import gensim
from gensim.models import word2vec
import multiprocessing
import jieba,re
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, confusion_matrix
from tensorflow.keras import regularizers

K=5#五折交叉验证
#reposts_count-->转发数;screen_name(user_name)-->用户昵称;followers_count-->粉丝数;friends_count-->关注数
#statuses_count-->微博数;favourites_count-->收藏数;created_at-->创建时间;bi_followers_count;互粉数
def find_question_sign(text):
    result=0
    if re.search(pattern="(真的?|真的？|求证|真的假的|真的吗|未经证实)",string=text)!=None:
        result+=1
    if re.search(pattern="([那|这|它]不是真的|假的)",string=text)!=None:
        result+=1
    if re.search(pattern="(谣言|揭穿)",string=text)!=None:
        result+=1
    if re.search(pattern="(什么[?!][?1]*)",string=text)!=None:
        result+=1
    if re.search(pattern="([这|那|它]是真的吗)",string=text)!=None:
        result+=1
    return result
social_feature_nums=23

embeddingSize=128
doc_embed_dim=128
group_doc_embed_dim=200
class Config(object):
    def __init__(self):     # 初始化训练配置
        # 数据集路径
        self.dataSource = ""
        self.stopWordSource = "./stopword.txt"
        self.miniFreq = 1
        self.sequenceLength = 200
        self.batchSize = 32
        self.epochs = 15
        self.numClasses = 2
        self.rate = 0.8
        self.embeddingSize = embeddingSize
        self.dropoutKeepProb = 0.3
        # L2正则系数
        self.l2RegLambda = 0.1
        self.max_group_nums=10
        self.max_post_nums=50
        self.max_seq_len=30
        self.doc_embed_dim=doc_embed_dim
        self.group_doc_embed_dim=group_doc_embed_dim

def get_group_event_post(all_event_ids,all_event_path_dir,N=50,Min=5):
    #all_event_path_dir是一个文件夹，存储的是所有事件的每一个事件的所有post，以event_id.json命名
    S={}
    for event_id in tqdm(all_event_ids):
        event_str = str(event_id)
        tweets = json.load(open(os.path.join(all_event_path_dir, event_str + '.json'), "r", encoding='utf-8'))#tweets是这个事件的所有post集合
        text = []
        S[event_str]=[]
        n_i=len(tweets)#一个事件的所有post的数目，将一个事件的所有post至少分成Min组，每一组有N个post
        v,x,y=1,0,0
        while True:
            #例如n_i=275,int(n_i/50)==5
            if n_i>=N*Min:
                while v<=int(n_i/N):
                    x=N*(v-1)
                    y=N*v
                    S[event_str].append(tweets[x:y])
                    v+=1
                #此时S[event_str]存储的是[tweets[0:50],tweets[50:100],tweets[100:150],tweets[150:200],tweets[200:250]]
                #还剩下250--275一共25个post
                S[event_str].append(tweets[y:n_i])
            else:
                #例如n_i=174，174<50*5=250,int(n_i/5)==34
                while v<Min:
                    x=int(n_i/Min)*(v-1)
                    y=int(n_i/Min)*v
                    S[event_str].append(tweets[x:y])
                    v+=1
                #此时S[event_str]存储的是[tweets[0:34],tweets[34:68],tweets[68:102],tweets[102:136],tweets[136:170]]
                #还剩下170,171,172,173,174这五个post
                S[event_str].append(tweets[y:n_i])
            assert len(S[event_str])>=Min
            break
        #S是一个dict，key是所有事件的event_id，对应的values是该事件的所有的post的分组，至少Min组，每一组有N个post
    return S

def get_social_feature(interval_posts):
    #interval_posts是一个list，每一个值是一个dict，代表一个post
    bi_followers_count_list = []
    reposts_count_list = []
    friends_count_list = []
    followers_count_list = []
    statuses_count_list = []
    favourites_count_list = []
    comments_count_list = []
    text_length_list = []
    attitudes_count_list = []
    user_activeness=[]
    user_reputation=[]
    number_of_posts_per_day=[]
    number_of_followees_per_day=[]

    contain_symbal_1_count = 0
    contain_symbal_2_count = 0
    contain_symbal_3_count = 0
    contain_symbal_4_count = 0
    contain_symbol_5_count=0

    contain_verified_count = 0
    contain_user_description_count = 0
    contain_reposts_count = 0
    contain_url_count = 0
    tweets=interval_posts
    question_sign_count=0



    for tweet in tweets:
        # average of bi_followers_count、friends_count、followers_count、statuses_count、
        # average of favourites_count、comments_count、text_length
        timeStamp=tweet["user_created_at"]
        str_time=time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(timeStamp))
        register_day=(2020-int(str_time.split('--')[0]))*365+(13-int(str_time.split('--')[1]))*30

        bi_followers_count = tweet['bi_followers_count']
        bi_followers_count_list.append(bi_followers_count)

        friends_count = tweet['friends_count']
        friends_count_list.append(friends_count)#关注数

        followers_count = tweet['followers_count']
        followers_count_list.append(followers_count)#followers_count是粉丝数
        number_of_followees_per_day.append(followers_count/register_day)

        statuses_count = tweet['statuses_count']
        statuses_count_list.append(statuses_count)
        number_of_posts_per_day.append(statuses_count/register_day)

        user_reputation.append(followers_count/(friends_count+1))
        user_activeness.append(friends_count/(followers_count+1))

        favourites_count = tweet['favourites_count']
        favourites_count_list.append(favourites_count)

        comments_count = tweet['comments_count']
        comments_count_list.append(comments_count)

        text = tweet['text']
        text_length_list.append(len(text))
        if 'http' in text:
            contain_url_count += 1

        if find_question_sign(text)>0:
            question_sign_count+=1

        attitudes_count = tweet['attitudes_count']
        attitudes_count_list.append(attitudes_count)

        # ? ! @ #
        symbal_1 = text.count('？') + text.count('?')
        if symbal_1 > 0:
            contain_symbal_1_count += 1

        symbal_2 = text.count('!') + text.count('!')
        if symbal_2 > 0:
            contain_symbal_2_count += 1

        symbal_3 = text.count('@')
        if symbal_3 > 0:
            contain_symbal_3_count += 1

        symbal_4 = text.count('#')
        if symbal_4 > 0:
            contain_symbal_4_count += 1

        symbol_5=text.count('？！')+text.count('?!')+text.count('?！')+text.count('？!')
        if symbol_5>0:
            contain_symbol_5_count+=1

        # verified, user_description, reposts_count
        verified = tweet['verified']
        if verified == True:
            contain_verified_count += 1

        user_description = tweet['user_description']
        if len(user_description) > 0:
            contain_user_description_count += 1

        reposts_count = tweet['reposts_count']
        if reposts_count > 0:
            contain_reposts_count += 1#转发数量这个特征简单的认为有无转发不太合理，转发数量多的post
                #明显其text的真实性和虚假性都很高

        #注意到一个事件有多少个相关的post，就有多少个user，因此我们取平均
    average_bi_followers = np.mean(bi_followers_count_list)
    average_friends = np.mean(friends_count_list)
    average_followers = np.mean(followers_count_list)
    average_statuses = np.mean(statuses_count_list)#平均的微博数量
    average_favourites = np.mean(favourites_count_list)
    average_comments = np.mean(comments_count_list)
    average_text_length = np.mean(text_length_list)
    average_attitudes = np.mean(attitudes_count_list)

    average_number_of_posts_per_day=np.mean(number_of_posts_per_day)
    average_number_of_followees_per_day=np.mean(number_of_followees_per_day)
    average_user_reputation=np.mean(user_reputation)
    average_user_activateness=np.mean(user_activeness)

    symbal_1_fraction = contain_symbal_1_count / len(tweets)
    symbal_2_fraction = contain_symbal_2_count / len(tweets)
    symbal_3_fraction = contain_symbal_3_count / len(tweets)
    symbal_4_fraction = contain_symbal_4_count / len(tweets)
    symbol_5_fraction=contain_symbol_5_count/len(tweets)

    question_sign_fraction=question_sign_count/len(tweets)

    verified_fraction = contain_verified_count / len(tweets)#How many users have been verified of those relevent posts
    user_description_fraction = contain_user_description_count / len(tweets)#所有relevent post's user有多少有desctiption
    reposts_fraction = contain_reposts_count / len(tweets)#有很多的帖子是转发微博，所以reposts_count=0，我认为
        #reposts的数目应该取所有的post中reposts_count最大的值
    url_fraction = contain_url_count / len(tweets)

    social_feature = [len(tweets), average_bi_followers, average_friends, average_followers, average_statuses,average_favourites,
              average_comments, average_text_length, average_attitudes,symbal_1_fraction, symbal_2_fraction, symbal_3_fraction,
              symbal_4_fraction, symbol_5_fraction,verified_fraction, user_description_fraction, reposts_fraction, url_fraction,question_sign_fraction,average_number_of_followees_per_day,average_number_of_posts_per_day,
              average_user_activateness,average_user_reputation]
    return social_feature#23个特征

def get_input_data(all_event_group_post,event2label):
	#该函数返回值有两个，shape分别为[num_events,组数，组内post数目],[num_events,组数,social_feature_nums]
	input_text_data=[]
	input_social_feature_data=[]
	event_label=[]
	for event_id,event_group_post in tqdm(all_event_group_post.items()):
		#event_group_post是一个list，长度不固定，最少为Min
		current_group_text_data=[]#用来记录每一组的text
		current_group_social_feature=[]
		for interval_i in range(len(event_group_post)):
			current_interval_posts=event_group_post[interval_i]
			#current_interval_posts是一个list，长度为N，代表该interval内的所有post集合，每一个值是一个dict，代表一个post
			current_interval_text=[]#用来记录每一组内的所有post的text
			if len(current_interval_posts)==0:
				continue
			for post in current_interval_posts:
				current_interval_text.append(post['text'])
			current_group_text_data.append(current_interval_text)
			current_group_social_feature.append(get_social_feature(current_interval_posts))
			#get_social_feature返回的是长度为social_feature_nums的list
		input_text_data.append(current_group_text_data)
		input_social_feature_data.append(current_group_social_feature)
		event_label.append(int(event2label[str(event_id)]))
	return input_text_data,input_social_feature_data,event_label

def getWordEmbedding(words,model,embeddingSize=embeddingSize):
    vocab = []
    wordEmbedding = []
    vocab.append("pad")
    wordEmbedding.append(np.zeros(embeddingSize))
    vocab.append("UNK")
    wordEmbedding.append(np.random.randn(embeddingSize))
    unk_word=0
    print(words[:50])
    for word in words:
        try:
            vector = model[word]#注意顺序不能串了，vocab.append(word)不能在前
            vocab.append(word)
            wordEmbedding.append(vector)
        except:
            #print(word + " : 不存在于词向量中")
            unk_word+=1
    print("unk_word nums is ",unk_word)
    return vocab, np.array(wordEmbedding)


def getVocabulary(input_text_data,stopWordPath=None,miniFreq=0,pretrained_word2vec_path=None,pretrained_doc2vec_post_path=None,
                pretrained_doc2vec_group_path=None):
    stopWordList=[]
    if stopWordPath is not None:
        assert os.path.exists(stopWordPath)==True
        stopWordList=open(stopWordPath,encoding='gbk').read().split('\n')
    allWords=[]
    allSentences=[]
    group_sentence_for_doc2vec=[]
    #input_text_data存储的是所有事件的post
    for group_text_data in input_text_data:
        #group_text_data的长度是该事件被划分的组数目，所以group_text_data的每一个值是一个组内的所有post,将这些post看作一个整体训练
        #group_doc2vec
        #group_text_data是一个二维list，长度至少为5，每一个值是一个list,这个list的的每一个值是一个string
        for posts_text_data in group_text_data:
            group_sentence_for_doc2vec.append(jieba.lcut("".join(str(posts_text_data)[1:-1].replace("'",' ').split())))
            for text in posts_text_data:
                allWords.extend(jieba.lcut(text))
                allSentences.append(jieba.lcut(text))
    #group_sentence_for_doc2vec length is num_events and second dim is group_nums
    subWords = [word for word in allWords if word not in stopWordList]
    wordCount = Counter(subWords)  # 统计词频，排序
    sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
    print(list(sortWordCount)[:50])
    words = [item[0] for item in sortWordCount if item[1] >= miniFreq]
    print("length of words",len(words))
    if os.path.exists(pretrained_word2vec_path)==False:
        #allSentences是一个二维list，每一个值是一个list，是jieba分词后的句子，每一个值是一个word
        #在训练doc2vec_model时，我们也希望传入的是一个二维list，每一个值是一个list，是jieba分词后的句子，每一个值是一个word
        #但是此时的句子是一个group内的所有post连起来的句子
        model=word2vec.Word2Vec(allSentences,size=embeddingSize,min_count=2,window=5,workers=multiprocessing.cpu_count(), sg=1,iter=20)
        model.save(pretrained_word2vec_path)
        doc2vec_sentences=[]
        for i,sentence in enumerate(allSentences):
            doc2vec_sentences.append(gensim.models.doc2vec.TaggedDocument(words=sentence,tags=[i]))
        doc_post_model=gensim.models.doc2vec.Doc2Vec(documents=doc2vec_sentences,dm=0,vector_size=doc_embed_dim,workers=multiprocessing.cpu_count())
        doc_post_model.train(documents=doc2vec_sentences,total_examples=doc_post_model.corpus_count,epochs=20)
        doc_post_model.save(pretrained_doc2vec_post_path)

        group_doc2vec_sentences=[]
        for i,sentence in enumerate(group_sentence_for_doc2vec):
            group_doc2vec_sentences.append(gensim.models.doc2vec.TaggedDocument(words=sentence,tags=[i]))
        doc_group_model=gensim.models.doc2vec.Doc2Vec(documents=group_doc2vec_sentences,dm=0,vector_size=group_doc_embed_dim,window_size=20,workers=multiprocessing.cpu_count())
        doc_group_model.train(documents=group_doc2vec_sentences,total_examples=doc_group_model.corpus_count,epochs=20)
        doc_group_model.save(pretrained_doc2vec_group_path)
            
    else:
        print("Load pretrained word2vec model from ",pretrained_word2vec_path)
        model=gensim.models.Word2Vec.load(pretrained_word2vec_path)
        print("Load pretrained doc2vec model from ",pretrained_doc2vec_post_path)
        doc_post_model=gensim.models.Doc2Vec.load(pretrained_doc2vec_post_path)
        print("Load pretrained group doc2vec model from ",pretrained_doc2vec_group_path)
        doc_group_model=gensim.models.Doc2Vec.load(pretrained_doc2vec_group_path)
        
    vocab, wordEmbedding = getWordEmbedding(words,model)
    word2id=dict(zip(vocab, list(range(len(vocab)))))
    return word2id,wordEmbedding,doc_post_model,doc_group_model

def get_sentence_ids(tokens,word2id,max_seq_len):
	sentence_ids=[word2id.get(word,word2id['UNK']) for word in tokens]
	if len(sentence_ids)<max_seq_len:
		sentence_ids+=[0]*(max_seq_len-len(sentence_ids))
	return sentence_ids

def get_input_feature(input_text_data,input_social_feature_data,event_label,word2id,doc_post_model,doc_group_model,
                    max_group_nums=10,max_post_nums=50,max_seq_len=30,doc_dim=doc_embed_dim,group_doc_dim=group_doc_embed_dim):
    num_events=len(input_text_data)
    input_text_features=np.zeros((num_events,max_group_nums,max_post_nums,max_seq_len),dtype='int32')
    input_social_features=np.zeros((num_events,max_group_nums,social_feature_nums))
    input_post_embeddings=np.zeros((num_events,max_group_nums,max_post_nums,doc_dim))
    input_group_embeddings=np.zeros((num_events,max_group_nums,group_doc_dim))

    for i,each_event_social_feature in enumerate(input_social_feature_data):
        if len(each_event_social_feature)>max_group_nums:
            each_event_social_feature=each_event_social_feature[:max_group_nums]
        for j,group_social_feature in enumerate(each_event_social_feature):
            assert len(group_social_feature)==social_feature_nums
            input_social_features[i][j]=np.array(group_social_feature)

    for i,group_text_data in tqdm(enumerate(input_text_data)):
        if len(group_text_data)>max_group_nums:
            group_text_data=group_text_data[:max_group_nums]
        for j,posts_text_data in enumerate(group_text_data):
            if len(posts_text_data)>max_post_nums:
                posts_text_data=posts_text_data[:max_post_nums]
            group_tokens=jieba.lcut("".join(str(posts_text_data)[1:-1].replace("'",' ').split()))
            input_group_embeddings[i][j]=doc_group_model.infer_vector(group_tokens,alpha=0.01,steps=20)
            for k,text in enumerate(posts_text_data):
                tokens=jieba.lcut(text)
                text_embed=doc_post_model.infer_vector(tokens,alpha=0.01,steps=10)#vector
                input_post_embeddings[i][j][k]=text_embed
                if len(tokens)>max_seq_len:
                    tokens=tokens[:max_seq_len]
                sentence_ids=get_sentence_ids(tokens,word2id,max_seq_len)
                input_text_features[i][j][k]=np.array(sentence_ids,dtype="int32")
    return [input_text_features,input_social_features,input_post_embeddings,input_group_embeddings]

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        #assert len(input_shape) == 3#(None,seq_length,dim)
        self.W = self.add_weight(name='att_weight',
                                 shape=(int(input_shape[1]), int(input_shape[1])),
                                 initializer='uniform',
                                 trainable=True)#(seq_length,seq_length)
        self.b = self.add_weight(name='att_bias',
                                 shape=(int(input_shape[1]),),
                                 initializer='uniform',
                                 trainable=True)#(seq_length,)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (None,seq_length,dim)
        x = backend.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (None, dim,seq_length)
        a = backend.softmax(backend.tanh(backend.dot(x, self.W) + self.b))
        #(None,dim,seq_length)*(seq_length,seq_length)+(seq_length,)
        #a.shape==(None,dim,seq_length)
        outputs = backend.permute_dimensions(a * x, (0, 2, 1))
        #(None,seq_length,dim)
        outputs = backend.sum(outputs, axis=1)#(None,dim)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

def train_lstm(n_symbols,embedding_matrix,config):
    #(batch_size,max_group_nums,max_post_nums,max_seq_len)
    max_group_nums=config.max_group_nums
    max_post_nums=config.max_post_nums
    max_seq_len=config.max_seq_len
    main_input=Input(shape=(max_group_nums,max_post_nums,max_seq_len))
    sub_input=Input(shape=(max_group_nums,social_feature_nums))
    doc_embed_input=Input(shape=(max_group_nums,max_post_nums,config.doc_embed_dim))
    group_doc_embed_input=Input(shape=(max_group_nums,config.group_doc_embed_dim))
    
    if n_symbols!=embedding_matrix.shape[0]:
        assert n_symbols==embedding_matrix.shape[0]+1
        n_symbols=embedding_matrix.shape[0]
        print("change n_symbols!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    embedding_layer_main=Embedding(input_dim=n_symbols,output_dim=config.embeddingSize,weights=[embedding_matrix],
                                    input_length=max_seq_len,mask_zero=True)(main_input)
    dropout_layer_1 = Dropout(config.dropoutKeepProb)(embedding_layer_main)
    #shape==(batch_size,max_group_nums,max_post_nums,max_seq_len,embeddingSize)
    bid_GRU_layer_1=Bidirectional(GRU(32,activation="tanh",recurrent_dropout=0.5,return_sequences=True),
                merge_mode='concat')(backend.reshape(dropout_layer_1,shape=[-1,max_seq_len,config.embeddingSize]))
    #shape==(batch_size*max_group_nums*max_post_nums,max_seq_len,64)
    bn_layer_1 = BatchNormalization()(bid_GRU_layer_1)
    attention_layer_1 = AttentionLayer()(bn_layer_1)#(batch_size*max_group_nums*max_post_nums,64)
    
    attention_layer_1=backend.reshape(attention_layer_1,shape=[-1,max_post_nums,64])
    doc_embed_input_=backend.reshape(doc_embed_input,shape=[-1,max_post_nums,config.doc_embed_dim])
    doc_embed=Dense(64,activation="tanh")(doc_embed_input_)
    merge_doc_embed=concatenate([doc_embed,attention_layer_1],axis=2)
    bid_GRU_layer_2 = Bidirectional(GRU(32, activation='tanh', dropout=0.5, recurrent_dropout=0.5, return_sequences=True),
                merge_mode='concat')(merge_doc_embed)
    #shape==(batch_size*max_group_nums,max_post_nums,64)
    bn_layer_2 = BatchNormalization()(bid_GRU_layer_2)
    attention_layer_2 = AttentionLayer()(bn_layer_2)#(batch_size*max_group_nums,64)

    attention_layer_2=backend.reshape(attention_layer_2,shape=[-1,max_group_nums,64])

    group_doc_embed_input_=backend.reshape(group_doc_embed_input,shape=[-1,max_group_nums,config.group_doc_embed_dim])
    group_doc_embed=Dense(128,activation="tanh")(group_doc_embed_input_)
    merge_group_doc_embed=concatenate([group_doc_embed,attention_layer_2],axis=2)#(batch_size,max_group_nums,64+128)
    bid_GRU_layer_3 = Bidirectional(GRU(32, activation='tanh', dropout=0.5, recurrent_dropout=0.5, return_sequences=True),
                merge_mode='concat')(merge_group_doc_embed)

    bn_layer_3 = BatchNormalization()(concatenate([bid_GRU_layer_3,sub_input],axis=2))
    attention_layer_3 = AttentionLayer()(bn_layer_3) #(batch_size,64+social_feature_nums)

    dense_layer_1=Dense(64,activation="tanh")(sub_input)#(batch_size,max_group_nums,64)
    social_attention_layer=AttentionLayer()(dense_layer_1)#(batch_size,64)

    merge_layer=concatenate([attention_layer_3,social_attention_layer],axis=1)#(bathc_size,128+social_attention_layer)
    dropout_layer=Dropout(config.dropoutKeepProb)(merge_layer)
    output_layer=Dense(2,activation='softmax')(Dense(32,activation="tanh")(dropout_layer))

    model=Model([main_input,sub_input,doc_embed_input,group_doc_embed_input],output_layer)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

Weibo_file="./data/Weibo.txt"
Weiboeids_train_file="./data/train_event_ids.json"
Weiboeids_test_file="./data/test_event_ids.json"
all_post_file="./Weibo/"
stop_word_file="./stopword.txt"
if __name__=='__main__':
    f = open(Weibo_file, 'r')
    lines = f.readlines()
    f.close()
    event2label = {}#key是该事件的event_id，对应的value是该事件的label
    config=Config()
    for line in lines:
        line = line.replace('\t', ' ')
        line = line.split(' ')
        line.remove('\n')
        label = line[1][-1]
        eid = line[0][4:]
        event2label[eid] = label
    print("有%d个事件"%len(event2label))
    all_event_path_dir = all_post_file#all_event_path_dir是一个文件夹，存储的是所有事件的每一个事件的所有post，以event_id.json命名
    all_event_ids = json.load(open(Weiboeids_train_file, 'r'))['train']#存储的是所有事件的event_id

    test_all_event_ids=json.load(open(Weiboeids_test_file))["test"]

    print("%d events for train and %d events for test"%(len(all_event_ids),len(test_all_event_ids)))
    #assert len(all_event_ids)+len(test_all_event_ids)==len(event2label)
    all_event_group_post=get_group_event_post(all_event_ids,all_event_path_dir,N=50,Min=5)
    test_event_group_post=get_group_event_post(test_all_event_ids,all_event_path_dir,N=50,Min=5)
    #all_event_group_post是一个dict，key是所有事件的event_id，对应的values是该事件的所有的post的分组，至少Min组，每一组有N个post
    #每一组可以看做一个时间段，接下来是利用all_event_group_post提取出每一个事件的每一时间段内的所有post的text以及social_feature
    input_text_data,input_social_feature_data,event_label=get_input_data(all_event_group_post,event2label)
    test_input_text_data,test_input_social_feature,test_event_label=get_input_data(test_event_group_post,event2label)

    #input_text_data-->[num_events,组数，组内post数目]，每一个值是一个string
    #input_social_feature_data-->[num_events,组数,social_feature_nums]
    #接下来根据这三个变量构造输入数据的特征，注意到组数以及各个组内的post数目是不一致的
    #目的是要构造出来形如[num_events,groups_num,posts_num,max_seq_len]以及[num_events,groups_num,social_feature_nums]的两个张量作为输入特征
    #对于组数小于groups_num要pad，post数目小于posts_num要pad，以及句子长度小于max_seq_len要pad

    if os.path.exists("./word2id_and_embedding.pkl"):
        print("load word2id and word embedding from ./ word2id and embedding")
        (word2id,wordEmbedding)=pickle.load(open("./word2id_and_embedding.pkl","rb"))
        doc_post_model=gensim.models.Doc2Vec.load("./doc2vec_post_Model")
        doc_group_model=gensim.models.Doc2Vec.load("./doc2vec_group_Model")
    else:
        word2id,wordEmbedding,doc_post_model,doc_group_model=getVocabulary(input_text_data,stopWordPath=stop_word_file,pretrained_word2vec_path='./word2vecModel',
                                        pretrained_doc2vec_post_path="./doc2vec_post_Model",
                                        pretrained_doc2vec_group_path="./doc2vec_group_Model")
        pickle.dump((word2id,wordEmbedding),open("./word2id_and_embedding.pkl","wb"))
    print('word embedding matrix shape : ',wordEmbedding.shape)
    #根据word2id将单词转为id
    input_features=get_input_feature(input_text_data,input_social_feature_data,event_label,word2id,doc_post_model,doc_group_model,
                                                                max_group_nums=10,max_post_nums=50,max_seq_len=30)
    #input_features==[input_text_features,input_social_features,input_post_embeddings,input_group_embeddings]
    [input_text_features,input_social_features,input_post_embeddings,input_group_embeddings]=input_features
    #input_post_embeddings.shape==(num_events,max_group_nums,max_post_nums,100)
    #input_group_embeddings.shape==(num_events,max_post_nums,200)
    #input_text_features.shape==(num_events,max_group_nums,max_post_nums,max_seq_len)
    #input_social_features.shape==(num_events,max_group_nums,social_feature_nums)
    target=np.array(event_label,dtype=np.float32).reshape((len(event_label),1))#(num_events,1)
    #assert len(target)==len(input_social_features)==len(all_event_ids)
    #根据打乱的data_index索引来随机的打乱各个事件
    
    ########################################################################################
    test_input_features=get_input_feature(test_input_text_data,test_input_social_feature,test_event_label,word2id,
                                        doc_post_model,doc_group_model,max_group_nums=10,max_post_nums=50,max_seq_len=30)
    [test_input_text_features,test_input_social_features,test_input_post_embeddings,test_input_group_embeddings]=test_input_features
    test_target=np.array(test_event_label,dtype=np.float32).reshape((len(test_event_label),1))
    ########################################################################################
    
    #由于已经将所有的数据随机打乱划分，所以这里不需要再随机打乱，只需要进行K份的分割
    train_index=int(len(target)/K)#当前是第二次验证，所以第二份留作验证,剩下的K-1份留作train

    print("train index------------>",train_index)
    trainTextReviews=np.concatenate([input_text_features[0:train_index],input_text_features[train_index*2:]],0).astype("int64")
    train_social_features=np.concatenate([input_social_features[0:train_index],input_social_features[train_index*2:]],0)
    trainLabels=to_categorical(np.concatenate([target[0:train_index],target[train_index*2:]],0),num_classes=2)

    evalTextReviews=input_text_features[train_index:train_index*2].astype('int64')
    eval_social_features=input_social_features[train_index:train_index*2]
    evalLabels=to_categorical(target[train_index:train_index*2],num_classes=2)

    testTextReviews=test_input_text_features
    test_social_features=test_input_social_features
    testLabels=to_categorical(test_target,num_classes=2)

    print("train text shape: ",trainTextReviews.shape,'train social_feature shape: ',train_social_features.shape,'train label shape: ',trainLabels.shape)
    print("eval text shape: ",evalTextReviews.shape,'eval social_feature shape: ',eval_social_features.shape,'eval label shape: ',evalLabels.shape)
    print("test text shape: ",testTextReviews.shape,'test social_feature shape: ',test_social_features.shape,'test label shape: ',testLabels.shape)

    ###########################################################################################
    if input_post_embeddings is not None:
        train_doc_embeddings=np.concatenate([input_post_embeddings[0:train_index],input_post_embeddings[train_index*2:]],0)
        eval_doc_embeddings=input_post_embeddings[train_index:train_index*2]
        test_doc_embeddings=test_input_post_embeddings
        print("train doc embeddings shape : ",train_doc_embeddings.shape)
        print("eval doc embeddings shape : ",eval_doc_embeddings.shape)
        print("test doc embeddings shape : ",test_doc_embeddings.shape)
    if input_group_embeddings is not None:
        train_group_doc_embeddings=np.concatenate([input_group_embeddings[:train_index],input_group_embeddings[train_index*2:]],0)
        eval_group_doc_embeddings=input_group_embeddings[train_index:train_index*2]
        test_group_doc_embeddings=test_input_group_embeddings
        print("train group_doc embeddings shape : ",train_group_doc_embeddings.shape)
        print("eval group_doc embeddings shape : ",eval_group_doc_embeddings.shape)
        print("test group_doc embeddings shape : ",test_group_doc_embeddings.shape)

    ###########################################################################################
    model=train_lstm(n_symbols=len(word2id)+1,embedding_matrix=wordEmbedding,config=config)
    model.summary()

    logdir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, mode='auto')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('./save_dir2/model_{epoch:02d}-{val_accuracy:.2f}.hdf5',
                                       save_best_only=True, save_weights_only=True)
    # reduce_lr,early_stopping,
    history = model.fit([trainTextReviews,train_social_features,train_doc_embeddings,train_group_doc_embeddings],trainLabels,epochs=15,batch_size=config.batchSize,
                    validation_data=([evalTextReviews,eval_social_features,eval_doc_embeddings,eval_group_doc_embeddings],evalLabels),shuffle=True,
                    callbacks=[early_stopping,reduce_lr,model_checkpoint])

    print(history.history.keys())
    predicts=model.predict([testTextReviews,test_social_features,test_doc_embeddings,test_group_doc_embeddings])#shape==(num_test_examples,2)
    y_predict_label=[]
    for each_predict in predicts:
        predict=list(each_predict)#len==2
        y_predict_label.append(predict.index(max(predict)))

    y_golden_label=[]
    for each_target in testLabels:
        golden_label_list=list(each_target)
        y_golden_label.append(golden_label_list.index(max(golden_label_list)))

    tn,fp,fn,tp=confusion_matrix(y_golden_label,y_predict_label).ravel()
    #tn==true negative(标签为0的样本正确的分类为0) fp==false positive(标签为0的样本错误的分类为1)
    #fn==false negative(标签为1的样本错误的分类为0) tp==true positive(标签为1的样本正确的分类为1)
    precision_real=tp/(tp+fp)#预测的为real的样本中有多少是正确预测的
    precision_fake=tn/(fn+tn)#预测的为fake的样本中有多少是正确预测的

    recall_real=tp/(tp+fn)#有多少的real样本被模型正确分类出来
    recall_fake=tn/(fp+tn)#有多少的fake样本被模型正确分类出来

    fscore_real=2*precision_real*recall_real/(precision_real+recall_real)
    fscore_fake=2*precision_fake*recall_fake/(precision_fake+recall_fake)

    accuracy=(tp+tn)/(tp+tn+fp+fn)

    print("precision_real: ",precision_real)
    print("precision_fake: ",precision_fake)
    print("recall_real: ",recall_real)
    print("recall_fake: ",recall_fake)
    print("fscore_real: ",fscore_real)
    print("fscore_fake: ",fscore_fake)
    print("accuracy: ",accuracy)



