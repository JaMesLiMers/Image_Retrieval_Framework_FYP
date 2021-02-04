from typing import OrderedDict
import numpy as np
from Models.Bm25LMIR.source.lmir_bm25 import BM25_LMIR

class lmirBm25Model():
    """BM25 and LMIR algorithm model.

        调用forward或者直接调用这个类以获得结果, 包装并实现了混合查询. 
        
        在查询的过程中, 对输入的语句进行了混合权重查询.
        
        BM25和LMIR算法的合集类, 真实是基于lmir_bm25类的, 调用了这个类的下面几个方法:

            BM25_LMIR.BM25(query_tokens)
            BM25_LMIR.jelinek_mercer(query_tokens)
            BM25_LMIR.dirichlet(query_tokens)
            BM25_LMIR.absolute_discount(query_tokens)

    """

    def __init__(self, corpora, modelWeight=[0.25, 0.25, 0.25, 0.25], k1=1.5, b=0.75, lamb=0.1, mu=2000, delta=0.7, epsilon=0.25):
        """
        初始化BM25-LMIR的模型以及需要的参数.
        Input:
            corpora: list 分好词的句子.
            modelWeight: list 和为1的四个weight, 分别对应 [BM25, JM, DIR, ABS] 的weight.
            超参数:
                k1:         (BM25: algorithm)
                b:          (BM25: algorithm)
                lamb:       (LMIR.JM: algorithm)
                mu:         (LMIR.DIR: algorithm)
                delta:      (LMIR.ABS: algorithm)
                epsilon:    (BM25: idf none negative)
        """
        self.b = b
        self.k1 = k1
        self.mu = mu
        self.lamb = lamb
        self.delta = delta
        self.epsilon = epsilon

        self.model = BM25_LMIR(corpora, k1=self.k1, b=self.b, lamb=self.lamb, mu=self.mu, delta=self.delta, epsilon=self.epsilon)
        self.modelList = [self.model.BM25, self.model.jelinek_mercer, self.model.dirichlet, self.model.absolute_discount]
        self.modelWeight = modelWeight

    def standardization(self, data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma if sigma != 0 else data

    def forward(self, X):
        """
        查询部分, 由于可以整个句子输入, 所以不加weight.

        Input:
            X: list 分好词的查询部分, 一个元素是一个词, 多个词组合在一起查询.
        
        Return:
            dict{
                “模型名”: [结果nparray]
                “ALL”: [加权结果nparray]
            }
        """

        # for BM25 (larger is better)
        bm25Result = np.array(self.model.BM25(X))

        bm25ResultNorm = self.standardization(bm25Result)

        # for lmir (smaller is better)
        # for long queries
        lmirJmResult = np.array(self.model.jelinek_mercer(X))
        lmirJmResultNorm = -self.standardization(lmirJmResult)
        # for short queries
        lmirDirResult = np.array(self.model.dirichlet(X))
        lmirDirResultNorm = -self.standardization(lmirDirResult)
        # less efficent
        lmirAbsResult = np.array(self.model.absolute_discount(X))
        lmirAbsResultNorm = -self.standardization(lmirAbsResult)

        weightResult = bm25ResultNorm * self.modelWeight[0] + \
                    lmirJmResultNorm * self.modelWeight[1] + \
                    lmirDirResultNorm * self.modelWeight[2] + \
                    lmirAbsResultNorm * self.modelWeight[3]

        weightResult = self.standardization(weightResult)

                
        return {
            "BM25": bm25ResultNorm,
            "JM": lmirJmResultNorm,
            "DIR": lmirDirResultNorm,
            "ABS": lmirAbsResultNorm,
            "ALL": weightResult
        }

    def forwardWords(self, X, weight):
        """
        查询部分, 输入为一系列的关键词, 无多余字符.

        Input:
            X: list 分好词的查询部分, 每一个元素是一个词.
            wordWeight: (list) 对于每一个词占算法重要性的比例, 在算法计算完成后最终融合时, 将会按照权重进行加权求和.
        Return:
            dict{
                "1": {
                        “模型名”: [结果nparray]
                        “ALL”: [加权结果nparray]
                        },
                "2": {
                        “模型名”: [结果nparray]
                        “ALL”: [加权结果nparray]
                        },
                "ALL": {
                        “模型名”: [结果nparray]
                        “ALL”: [加权结果nparray]
                        },
                }
        """
        allResult = OrderedDict()

        # cal every word
        for i in range(len(X)):
            wordResult = self.forward([X[i]])
            allResult[i] = wordResult

        def weightedSum(diction:dict, key, weight):
            """
            calculate part result in dict
            """
            tosum = []
            for k, v in diction.items():
                tosum.append(v[key]*weight[k])
        
            return np.sum(tosum, 0)
            
        # sum all word
        allResult["ALL"] = {
            "BM25": weightedSum(allResult, "BM25", weight),
            "JM": weightedSum(allResult, "JM", weight),
            "DIR": weightedSum(allResult, "DIR", weight),
            "ABS": weightedSum(allResult, "ABS", weight),
        }

        # sum all model
        weightResult = allResult["ALL"]["BM25"] * self.modelWeight[0] + \
                    allResult["ALL"]["JM"] * self.modelWeight[1] + \
                    allResult["ALL"]["DIR"] * self.modelWeight[2] + \
                    allResult["ALL"]["ABS"] * self.modelWeight[3]

        allResult["ALL"]["ALL"] = weightResult

        return allResult["ALL"]
            


            
if __name__ == "__main__":
    words = "Hello there London".split()
    weight = [0.3, 0.3, 0.3]

    corpus = [
        "Hello there good man !",
        "It is quite windy in London",
        "How is the weather today ?"
        ]   

    tokenized_corpus = [doc.split(" ") for doc in corpus]


    bm25 = lmirBm25Model(tokenized_corpus)

    # for every sentence
    result = bm25(words)

    # for every keyword
    result2 = bm25.forwardWord(words, weight)

    print(result)
            
