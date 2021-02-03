import jieba
import numpy as np
from numpy.core.records import array
from Dataloader.Arch.arch import Arch
from Models.Bm25LMIR.lmirBm25Model import lmirBm25Model

class archLmirBm25Model():
    def __init__(self, archPath, model_weight=[0.25, 0.25, 0.25, 0.25]):
        """
        初始化模型, 初始化一个: 
            lmirBM25Model

        Input:
            archPath: archtectureDataset的数据位置
            modelWeight: list of model weight [BM25, JM, DIR, ABS]
    
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

        self.model = lmirBm25Model(self.corporaList, modelWeight=self.model_weight)

    def standardization(self, data):
        return (data-np.min(data))/(np.max(data)-np.min(data))

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
        result = self.model.forward(to_search)

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
        result = self.model.forwardWords(to_search, weights)

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
    # init model
    model = archLmirBm25Model(archPath=archPath)
    # search a list of word/sentence
    index, copora, annoId, imageId, sortedResult = model.searchSentence(listWords=["图书馆"])
    # print result
    for i in range(10):
        print("{}'th most similar index: {}\n".format(i+1, index[i]))
        print("{}'th most similar annoId: {}\n".format(i+1, annoId[i]))
        print("{}'th most similar imageId: {}\n".format(i+1, imageId[i]))
        print("{}'th most similar corpus: {}\n".format(i+1, copora[i]))

        



