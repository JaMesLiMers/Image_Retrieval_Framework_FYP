# from re import match
import sys, os
sys.path.append('/Data_HDD/changliu/FYP/Image_Retrieval_Framework_FYP')

import jieba
from Dataloader.Arch.arch import Arch

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import Models.Word2Vec.source.w2v_tfidf as w2vModel
import numpy as np
import gensim
import time


class W2VModel:
    def __init__(self, corporaList, w2v_f, stopwords_f, pretrained_w2v=None, saved_corpora_vec_M=None):
        """实现使用语言模型对句子和句子, 词语对句子的匹配.

        初始化模型:
            StatisticModel(corpora)

            Args:
                corporaList: 多个语料组成的列表, 应该是一个嵌套Python列表. For example:
                    [["There", "is", "a", "cat"],
                     ["There", "is", "a", "dog"],
                     ["There", "is", "a", "wolf"]]
                w2v_f: str, 预训练的词向量文件路径
                stopwords_f: str, 停用词文件
                pretrained_w2v: gensim.models.KeyedVectors, 预训练word2vec模型 (调试用)
                saved_corpora_vec_M: str, 保存的文档向量文件
        """
        self.corpora = self.process_corpora(corporaList, stopwords_f)
        print('loading pre-trained w2v model...')
        tic = time.time()
        # self.w2v_model = gensim.models.KeyedVectors(vector_size=300)
        if pretrained_w2v:
            self.w2v_model = pretrained_w2v
        else:
            if w2v_f.endswith('.bin'):
                self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_f, binary=True)
            else:
                self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_f)
        print('Done (t={:0.2f}s)'.format(time.time()- tic))

        self.tfidf_model, self.tfidf_M, self.corpora_vocab = self.fit_tfidf()

        if saved_corpora_vec_M:
            self.corpora_vec_M = np.load(saved_corpora_vec_M)
        else: 
            self.corpora_vec_M = None

        self.W2V_TFIDF_Model = w2vModel.W2V_TFIDF(corpora=self.corpora, tfidf_model=self.tfidf_model, tfidf_M=self.tfidf_M, w2v_model=self.w2v_model, corpora_vocab=self.corpora_vocab)

    def process_corpora(self, corporaList, stopwords_f):
        """预处理语料库
        Args:
            corporaList: 多个语料组成的列表, 应该是一个嵌套Python列表. For example:
                [["There", "is", "a", "cat"],
                 ["There", "is", "a", "dog"],
                 ["There", "is", "a", "wolf"]]
            stopwords_f: txt路径, 中文停用词表。每一行为一个词.
        
        Return:
            w2v_corpora: list of string, 适用于word2vec和tfidf. For example:
                [["There is a cat"],
                 ["There is a dog"],
                 ["There is a wolf"]]
        """
        stopwords = open(stopwords_f, encoding='utf-8').read()
        corpora = []
        for text in corporaList:
            text_p = []
            for word in text:
                word = word.lower()  # important!!!
                if word in stopwords:
                    continue
                text_p.append(word)
            corpora.append(' '.join(text_p))
        return corpora

    def fit_tfidf(self):
        """构建tfidf模型

        Return:
            tfidf_model: sklearn.feature_extraction.text.TfidfVectorizer
        """
        tfidf_vectorier = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')  # token_pattern: 保留长度为1的词
        tfidf_M = tfidf_vectorier.fit_transform(self.corpora).toarray()
        corpora_vocab = tfidf_vectorier.get_feature_names()
        return tfidf_vectorier, tfidf_M, corpora_vocab

    def w2v_match(self, queryTokens):
        """tfidf加权的word2vec算法实现

        Args: 
            queryTokens: (list) 分好词的查询部分, 一个元素是一个词, 多个词组合在一起查询.

        Return:
            返回对每一个句子的BM25查询值归一化后的结果, corpora有几个句子, 结果中就应该有几个查询值. 格式如下:
                [0, 0, 0]
        
        """
        process_token = []
        for word in queryTokens:
            word = word.lower()  # important
            if word not in self.W2V_TFIDF_Model.w2v_vocab:
                continue
            process_token.append(word)
        process_token = ' '.join(process_token)
        token_vec = self.W2V_TFIDF_Model.token2vec(process_token)
        if self.corpora_vec_M is None:
            corpora_vec_M = self.W2V_TFIDF_Model.corpora2vec()
            self.corpora_vec_M = corpora_vec_M
        else:
            corpora_vec_M = self.corpora_vec_M

        sim_list = pairwise_distances(X=token_vec, Y=corpora_vec_M, metric="cosine").squeeze()
        return sim_list

    def CT_recom(self, id, top_n=10):
        """ Click Through Recommendation

        Args: 
            id: int, 选中的语料的id
            top_n: int, 返回的语料的数量

        Return:
            返回top_n个相似语料的id
        
        """
        if self.corpora_vec_M is not None:
            corpora_vec_M = self.corpora_vec_M
        else:
            corpora_vec_M = self.W2V_TFIDF_Model.corpora2vec()
            self.corpora_vec_M = corpora_vec_M

        query_vec = np.expand_dims(corpora_vec_M[id], 0)
        
        sim_list = pairwise_distances(X=query_vec, Y=corpora_vec_M, metric="cosine").squeeze()

        sim_ids = list(np.array(sim_list).argsort()[1:top_n])
        return sim_ids

    def standarlization(self, data):
        """归一化搜索结果

        Args:
            data: nparray, 保存的都是float
        Return:
            Python列表, 保存归一化后的相似度
        """
        data = np.array(data)
        minValue = np.min(data)
        maxValue = np.max(data)
        result = (data - minValue) / (maxValue - minValue) if (maxValue - minValue) != 0 else data
        return list(result)

    # def forward(self, X):
    #     """
    #     查询部分, 由于可以整个句子输入, 所以不加weight.

    #     Input:
    #         X: list 分好词的查询部分, 一个元素是一个词, 多个词组合在一起查询.
        
    #     Return:
    #         dict{
    #             “模型名”: [结果nparray]
    #             “ALL”: [加权结果nparray]
    #         }
    #     """
    #     sim_arr = self.w2v_match(X)
    #     sim_arr = self.standarlization(sim_arr)

    #     return {"ALL": sim_arr}


if __name__ == '__main__':
    archPath = "/Data_HDD/changliu/FYP/Image_Retrieval_Framework_FYP/Dataset/Arch/DemoData_20201228.json"
    stopwords_f = '/Data_HDD/changliu/FYP/Image_Retrieval_Framework_FYP/utils/Stop_words/hit_stopwords.txt'
    saved_corpora_vec_M = './Models/Word2Vec/API/corpora_vec_M.npy'
    w2v_f = '/Data_HDD/changliu/data/merge_sgns_bigram_char300.bin'

    stopwords = open(stopwords_f, encoding='utf-8').read()
    
    archDataset = Arch(annotationFile=archPath)
    archDataset.reverseCharForAllContext()

    # generate annotation and corpora list
    annIdList = []
    corporaList = []
    notCutCorporaList = []
    for i, (annotation, content) in enumerate(archDataset.anns.items()):
        annIdList.append(annotation)
        corporaList.append(content["cutConcateText"])
        notCutCorporaList.append(content["concateText"])

        cut_content = jieba.cut(content["concateText"], cut_all=True)
        content_list = []
        for word in cut_content:
            if word not in stopwords:
                content_list.append(word)
    
    # 实例化模型
    w2v = W2VModel(corporaList=corporaList, w2v_f=w2v_f, stopwords_f=stopwords_f, saved_corpora_vec_M=saved_corpora_vec_M)

    # 测试
    test_1 = ['现代', '博物馆', '平面图']
    print('\n\ntest 1:', test_1)
    print('result:')
    sim_list_1 = w2v.w2v_match(test_1)

    ind_1 = np.array(sim_list_1).argsort()
    for i, j in enumerate(ind_1):
        if i > 9:
            break
        print(j)
        print(notCutCorporaList[j])


    test_2 = ['室内', '设计']
    print('\n\ntest 2:', test_2)
    print('result:')
    sim_list_2 = w2v.w2v_match(test_2)

    ind_2 = np.array(sim_list_2).argsort()
    for i, j in enumerate(ind_2):
        if i > 9:
            break
        print(j)
        print(notCutCorporaList[j])
