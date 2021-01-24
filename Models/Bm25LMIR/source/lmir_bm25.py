import Models.Bm25LMIR.source.freq_feature as ff
from Models.Bm25LMIR.source.freq_feature import cal_corpus_tf, cal_corpus_tp, cal_idf_BM25, cal_all_corpus_tf


class BM25_LMIR:
    def __init__(self, corpora, k1=1.5, b=0.75, lamb=0.1, mu=2000, delta=0.7, epsilon=0.25):
        """Initialize the pram that BM25 and LMIR algorithm need.

        BM25和LMIR算法的合集类, 实现了以下几个词与句子的相似度衡量方法:

            BM25_LMIR.BM25(query_tokens)
            BM25_LMIR.jelinek_mercer(query_tokens)
            BM25_LMIR.dirichlet(query_tokens)
            BM25_LMIR.absolute_discount(query_tokens)
        
        Attribute:
            需要的特征:
                N: 语料库的总长度, 为int.              (BM25 & LMIR)
                all_tp: 所有语料中词的出现概率          (LMIR)
                all_idf: 所有语料中词的idf             (BM25)
                corpus_tf: 单个语料中的词的词频         (BM25 & LMIR)
                corpus_tp: 单个语料中的词的出现概率      (LMIR)
                corpus_length: 语料库各个句子的长度     (BM25 & LMIR)
                avg_doc_length: 语料库的平均文档长度    (BM25)

            超参数的定义:
                k1:         (BM25: algorithm)
                b:          (BM25: algorithm)
                lamb:       (LMIR.JM: algorithm)
                mu:         (LMIR.DIR: algorithm)
                delta:      (LMIR.ABS: algorithm)
                epsilon:    (BM25: idf none negative)

        """
        # Hyper parameter
        self.b = b
        self.k1 = k1
        self.mu = mu
        self.lamb = lamb
        self.delta = delta
        self.epsilon = epsilon

        # Feature pre-request
        self.N, \
        self.all_tp, \
        self.all_idf, \
        self.corpus_tf, \
        self.corpus_tp, \
        self.corpus_length, \
        self.avg_doc_length = self.cal_all_feature_for_BM25_lmir(corpora)

        return None


    def cal_all_feature_for_BM25_lmir(self, corpora):
        """计算BM25和LMIR任务需要的各种特征并返回

        LIMR任务需要的特征包括:
            N: 语料库的总长度, 为int.              (BM25 & LMIR)
            all_tp: 所有语料中词的出现概率          (LMIR)
            all_idf: 所有语料中词的idf             (BM25)
            corpus_tf: 单个语料中的词的词频         (BM25 & LMIR)
            corpus_tp: 单个语料中的词的出现概率      (LMIR)
            corpus_length: 语料库各个句子的长度     (BM25 & LMIR)
            avg_doc_length: 语料库的平均文档长度    (BM25)

        这个函数负责计算这些值.

        Args:
            corpora: 多个语料组成的列表, 应该是一个嵌套Python列表. For example:

                [["There", "is", "a", "cat"],
                ["There", "is", "a", "dog"],
                ["There", "is", "a", "wolf"]]

        Return:
            函数返回值, 分别对应:

            N: 语料库的总长度, 为int.
            all_tp: 所有语料中词的出现概率, 是一个Python字典, 总结了所有的语料中的词的出现频率特征.
            all_idf: 所有语料中词的idf, 是一个Python字典, 和上文相似, 总结了所有的语料中的词的出现频率特征.
            corpus_tf: 单个语料中的词的词频, 列表中每一个元素都是Python字典, 按位置对应每一条语料的词频特征.
            corpus_tp: 单个语料中的词的出现概率, 列表中每一个元素都是Python字典, 按位置对应每一条语料中的词的出现频率特征.
            corpus_length: 语料库各个句子的长度, 为Python List, 每一位置对应一个句子的长度.
            avg_doc_length: 语料库的平均文档长度, 公式为: 文本长度总量/语料库总长度

        """
        # init
        N = len(corpora)
        all_tp = []
        all_idf = {}
        corpus_tf = []
        corpus_tp = []
        all_tf_LMIR = {}
        all_tf_BM25 = {}
        corpus_length = []
        avg_doc_length = 0


        # cal length, term frequence, term prob for every corpus
        for i in corpora:
            tf = cal_corpus_tf(i)
            corpus_length.append(len(i))
            corpus_tf.append(tf)
            corpus_tp.append(cal_corpus_tp(tf))

        # cal all term freque
        all_tf_LMIR = cal_all_corpus_tf(corpus_tf)
        all_tf_BM25 = cal_all_corpus_tf(corpus_tf, sentence_wide=True)

        # cal all term prob & idf
        all_tp = cal_corpus_tp(all_tf_LMIR)
        all_idf = cal_idf_BM25(N, all_tf_BM25, epsilon=self.epsilon)

        # cal average doc length
        avg_doc_length = sum(corpus_length) / N

        return N, all_tp, all_idf, corpus_tf, corpus_tp, corpus_length, avg_doc_length

    def BM25(self, query_tokens):
        """Wrapper for BM25"""
        return ff.BM25(query_tokens, 
                       self.N, 
                       self.all_idf, 
                       self.corpus_tf, 
                       self.corpus_length, 
                       self.avg_doc_length, 
                       k1=self.k1, 
                       b=self.b)

    def jelinek_mercer(self, query_tokens):
        """Wrapper for LMIR.JM"""
        return ff.jelinek_mercer(query_tokens, 
                                 self.N, 
                                 self.all_tp, 
                                 self.corpus_tp, 
                                 lamb=self.lamb)
    
    def dirichlet(self, query_tokens):
        """Wrapper for LMIR.DIR"""
        return ff.dirichlet(query_tokens, 
                            self.N, 
                            self.all_tp, 
                            self.corpus_tf, 
                            self.corpus_length,
                            mu=self.mu)

    def absolute_discount(self, query_tokens):
        """Wrapper for LMIR.ABS"""
        return ff.absolute_discount(query_tokens, 
                                    self.N, 
                                    self.all_tp, 
                                    self.corpus_tf, 
                                    self.corpus_length, 
                                    delta=self.delta)


            

