import Models.StatisticModel.source.lmir_bm25 as statisticModel
import numpy as np


class StatisticModel:
    def __init__(self, corpora):
        """实现使用语言模型对句子和句子, 词语对句子的匹配.

        实现的模型有:
            BestMatch25算法
            LMIR.jelinekMercer算法
            LMIR.dirichlet算法
            LMIR.absoluteDiscount算法

        初始化模型:
            StatisticModel(corpora)

            Args:
                corpora: 多个语料组成的列表, 应该是一个嵌套Python列表. For example:
                [["There", "is", "a", "cat"],
                ["There", "is", "a", "dog"],
                ["There", "is", "a", "wolf"]]
            
        使用模型参考下面五个API:
            StatisticModel.bestMatch25()
            StatisticModel.jelinekMercer()
            StatisticModel.dirichlet()
            StatisticModel.absoluteDiscount()
        """
        self.sourceModel = statisticModel.BM25_LMIR(corpora)

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

    def bestMatch25(self, queryTokens):
        """信息检索Probabilistic Model经典算法BM25的算法实现:

        Args: 
            queryTokens: (list) 分好词的查询部分, 一个元素是一个词, 多个词组合在一起查询.

        Return:
            返回对每一个句子的BM25查询值归一化后的结果, corpora有几个句子, 结果中就应该有几个查询值. 格式如下:

                [0, 0, 0]
        
        """
        result = np.array(self.sourceModel.BM25(queryTokens))
        result = self.standarlization(result)
        return result

    def jelinekMercer(self, queryTokens):
        """信息检索Language Model经典算法JM的算法实现:

        Args: 
            queryTokens: (list) 分好词的查询部分, 一个元素是一个词, 多个词组合在一起查询.

        Return:
            返回对每一个句子的JM查询值归一化后的结果, corpora有几个句子, 结果中就应该有几个查询值. 格式如下:

                [0, 0, 0]
        
        """
        result = -np.array(self.sourceModel.jelinek_mercer(queryTokens))
        result = self.standarlization(result)
        return result

    def dirichlet(self, queryTokens):
        """信息检索Language Model经典算法Dir的算法实现:

        Args: 
            queryTokens: (list) 分好词的查询部分, 一个元素是一个词, 多个词组合在一起查询.

        Return:
            返回对每一个句子的Dir查询值归一化后的结果, corpora有几个句子, 结果中就应该有几个查询值. 格式如下:

                [0, 0, 0]
        
        """
        result = -np.array(self.sourceModel.dirichlet(queryTokens))
        result = self.standarlization(result)
        return result
        
    def absoluteDiscount(self, queryTokens):
        """信息检索Language Model经典算法Abs的算法实现:

        Args: 
            queryTokens: (list) 分好词的查询部分, 一个元素是一个词, 多个词组合在一起查询.

        Return:
            返回对每一个句子的Abs查询值归一化后的结果, corpora有几个句子, 结果中就应该有几个查询值. 格式如下:

                [0, 0, 0]
        
        """
        result = -np.array(self.sourceModel.absolute_discount(queryTokens))
        result = self.standarlization(result)
        return result

# TODO: 测试使用情况
if __name__ == '__main__':
    pass