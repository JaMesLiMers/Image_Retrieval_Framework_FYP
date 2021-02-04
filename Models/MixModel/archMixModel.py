import jieba
import numpy as np
from numpy.core.records import array
from Dataloader.Arch.arch import Arch
from Models.Bm25LMIR.lmirBm25Model import lmirBm25Model
from Models.Doc2Vec.doc2vecModel import doc2vecModel

class archMixModel():
    def __init__(self, archPath, modelPath, model_weight=[0.5, 0.5]):
        """
        初始化模型, 分别初始化两个: 
            1. lmirBM25
            2. Doc2Vec

        Input:
            archPath: archtectureDataset的数据位置
            modelPath: doc2vec训练好的模型位置
            modelWeight: list of model weight[staticModel, featureModel]
    
        """
        self.archDataset = Arch(annotationFile=archPath)
        self.archDataset.reverseCharForAllContext()

        # generate annotation and corpora list
        self.annIdList = []
        self.corporaList = []
        self.notCutCorporaList = []
        for annotation, content in self.archDataset.anns.items():
            self.annIdList.append(annotation)
            self.corporaList.append(content["cutConcateText"])
            self.notCutCorporaList.append(content["concateText"])

        # model weight [staticModel, featureModel]
        self.model_weight = model_weight

        self.staticModel = lmirBm25Model(self.corporaList)
        self.featureModel = doc2vecModel(self.corporaList, modelPath)


    def standardization(self, data):
        minValue = np.min(data)
        maxValue = np.max(data)
        return (data - minValue) / (maxValue - minValue) if (maxValue - minValue) != 0 else data

    def forward(self, X):
        """
        查询部分, 由于可以整个句子输入, 所以不加weight.

        Input:
            X: list 分好词的查询部分, 一个元素是一个词, 多个词组合在一起查询.
        
        Return:
            dict{
                “模型名”: [结果list]
                “ALL”: [加权结果list]
            }
        """
        resultStatic = self.staticModel.forward(X)["ALL"]
        resultFeature = self.featureModel.forward(X)["ALL"]

        resultWeighted = resultStatic * self.model_weight[0] + \
                         resultFeature * self.model_weight[1]

        resultWeighted = self.standardization(resultWeighted)

        return {
                "staticModel": resultStatic,
                "featureModel": resultFeature,
                "ALL": resultWeighted,
                }

    # TODO
    def forwardWords(self, X, weights):
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
        raise NotImplementedError

    def searchSentence(self, listWords):
        """
        Search a sentence contains keyword

        Input:
            listWords: 未分好词的查询部分, 是一个或几个字符串的数组
        """
        # cut words
        to_search = []
        for i in listWords:
            to_search += list(jieba.cut(i, True))

        # search
        result = self.forward(to_search)

        # get index
        index = np.argsort(result["ALL"])[::-1]
        sortedResult = self.standardization(np.sort(result["ALL"])[::-1])
        
        # result
        imageId = []
        annoId = []
        copora = []
        for i in index:
            annoId.append(self.annIdList[i])
            copora.append(self.notCutCorporaList[i])
            imageId.append(self.archDataset.anns[self.annIdList[i]]["imageId"])

        return sortedResult, index, copora, annoId, imageId

    def searchWords(self, listWords, weights):
        """
        Search a list of keyword

        Input:
            listWords: 未分好词的查询部分, 是一个或几个字符串的数组
        """
        # cut words
        to_search = []
        for i in listWords:
            to_search += list(jieba.cut(i, True))

        # search
        result = self.forwardWords(to_search, weights)

        # get index
        index = np.argsort(result["ALL"])[::-1]
        sortedResult = self.standardization(np.sort(result["ALL"])[::-1])
        
        # result
        imageId = []
        annoId = []
        copora = []
        for i in index:
            annoId.append(self.annIdList[i])
            copora.append(self.notCutCorporaList[i])
            imageId.append(self.archDataset.anns[self.annIdList[i]]["imageId"])

        # result
        return sortedResult, index, copora, annoId, imageId

if __name__ == "__main__":
    # how to use:
    archPath = "./Dataset/Arch/DemoData_20201228.json"
    modelPath = "./Models/Doc2Vec/source/trained_model/Arch/d2v_model.model"
    # init model
    model = archMixModel(archPath=archPath, modelPath=modelPath)
    # search a list of word/sentence
    sortedResult, index, copora, annoId, imageId = model.searchSentence(listWords=["图书馆", "现代", "平面图"])
    # print result
    for i in range(10):
        print("{}'th most similar index: {}\n".format(i+1, index[i]))
        print("{}'th most similar annoId: {}\n".format(i+1, annoId[i]))
        print("{}'th most similar imageId: {}\n".format(i+1, imageId[i]))
        print("{}'th most similar corpus: {}\n".format(i+1, copora[i]))
        
            