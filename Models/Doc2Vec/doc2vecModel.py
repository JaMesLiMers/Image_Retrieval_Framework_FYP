from typing import OrderedDict
import numpy as np
from Models.Doc2Vec.source.Doc2VecModel import Doc2VecModel

class doc2vecModel():
    """doc2vec algorithm model.

        调用forward或者直接调用这个类以获得结果, 包装并实现了查询. 
    """
    def __init__(self, corpora, model_path):
        """
        Input:
            corpora: list 分好词的句子.
            model_path: str 模型保存的位置.
        """
        self.corpora = corpora
        self.d2v = Doc2VecModel()
        self.trained_model = self.d2v.load_model(model_path)

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
        sim_arr = self.d2v.retrieve(test_text=X, model_dm=self.trained_model, corpus=self.corpora)
        sim_arr = self.standardization(sim_arr)

        return {"ALL": sim_arr}
