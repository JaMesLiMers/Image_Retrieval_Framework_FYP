import numpy as np
from numpy.core.fromnumeric import size
from tqdm import tqdm

class W2V_TFIDF:
    def __init__(self, corpora, tfidf_model, tfidf_M, w2v_model, corpora_vocab):
        """Initialize the pram that W2V_TFIDF algorithm need.
        W2V_TFIDF算法类, 实现了对词向量进行TFIDF加权得到句向量的相似度衡量方法。

        Args:
            corpora: 多个语料组成的列表, Python列表. For example:
                ["There is a cat",
                 "There is a dog",
                 "There is a wolf"]

            tfidf_model: sklearn.feature_extraction.text.TfidfVectorizer
                已经fit_transform(corpora)

            w2v_model: gensim.models.KeyedVectors

            corpora_vocab: corpora的所有词汇

        """
        self.corpora = corpora

        self.tfidf_model = tfidf_model
        self.tfidf_M = tfidf_M
        self.corpora_vocab = corpora_vocab

        self.w2v_model = w2v_model
        self.w2v_vocab = list(self.w2v_model.vocab)

    def corpora2vec(self):  # problem: 非常慢
        """将corpora的语料转为文档向量

        Return:
            doc_vec_M: np.array, shape: (语料数 * 词向量维数)
        """
        doc_vec_M = np.zeros((len(self.corpora), self.w2v_model.vector_size))   # 语料数 * 词向量维数
        
        corpora_vocab_index = {}
        for i, word in enumerate(self.corpora_vocab):
            corpora_vocab_index[word] = i

        # w2v_vocab_dict = {}
        # for i, word in enumerate(self.w2v_vocab):
        #     w2v_vocab_dict[word] = self.w2v_model.get_vector(word)

        for i, sample in enumerate(tqdm(self.corpora)):
            for word in sample.split(' '):
                if word not in self.w2v_vocab:  # 没有对应词向量
                    continue
                word_index = corpora_vocab_index[word]
                doc_vec_M[i] += self.tfidf_M[i][word_index] * self.w2v_model.get_vector(word)   # debug note...
        
        np.save('corpora_vec_M.npy', doc_vec_M)
        return doc_vec_M
        
    def token2vec(self, queryTokens):
        """tokens to vec

        Args:
            queryTokens: str, 分好词的查询部分, 一个元素是一个词, 以空格分隔, 多个词组合在一起查询.

        Return: 
            token_vec: np.array, shape: (1 * 词向量维数)
        """

        tfidf_vec = self.tfidf_model.transform([queryTokens]).toarray().squeeze()
        print(tfidf_vec.shape)
        print(self.tfidf_M.shape)

        corpora_vocab_index = {}
        for i, word in enumerate(self.corpora_vocab):
            corpora_vocab_index[word] = i

        token_vec = np.zeros((1, self.w2v_model.vector_size))   # 语料数 * 词向量维数
        for word in queryTokens.split(' '):
            if word not in self.w2v_vocab:  # 没有对应词向量
                continue
            word_index = corpora_vocab_index[word]
            token_vec[0] += tfidf_vec[word_index] * self.w2v_model.get_vector(word)
            print(tfidf_vec[word_index])
            print(tfidf_vec[word_index+1])
                
        return token_vec

# if __name__ == '__main__':
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     import gensim

#     corpora = ['两只 老虎 爱 跳舞', 
#                '小兔子 乖乖 拔 萝卜', 
#                '我 和 小鸭子 学 走路',
#                '童年 是 最美 的 礼物']

#     corpora_w2v = []
#     for sentence in corpora:
#         corpora_w2v.append(sentence.split(' '))
#     print(corpora_w2v)
#     vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
#     tfidf_model = vectorizer.fit_transform(corpora)
#     w2v_model = gensim.models.Word2Vec(corpora_w2v, size=300, min_count=1).wv
#     print('w2v_vocab:', list(w2v_model.vocab))
#     corpora_vocab = vectorizer.get_feature_names()
#     print('corpora_vocab:', corpora_vocab)
    
#     w2v_tfidf = W2V_TFIDF(corpora, tfidf_model, w2v_model, corpora_vocab)
#     result = w2v_tfidf.corpora2vec()
#     print(result)