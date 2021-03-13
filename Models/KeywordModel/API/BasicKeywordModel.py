from Models.KeywordModel.source.kwMatch import strictMatch, levenshteinDistance

class BasicKeywordModel:
    def __init__(self, corpora):
        """基本的基于编辑距离的模糊匹配方法

        实现的模型流程:
            首先匹配是否完全相同--有完全相同的Label直接返回.
            无完全相同的--则匹配指定编辑距离阀值的结果.

        初始化模型:
            BasicKeywordModel(corpora)

            Args:
                corpora: 多个关键词组成的列表, 应该是一个Python列表. 
                
                For example:
                    ["关键词1", "关键词2", "关键词3", ...]
            
        使用模型参考下面五个API:
            BasicKeywordModel.approxSearch()
        """
        self.corpora = corpora
        return None
    
    def approxSearch(self, keyWord, threshold=1):
        """搜索返回词库中与keyWord最接近的关键词

        1. 首先匹配是否完全相同的关键词, 有完全相同的关键词直接返回对应词.
        2. 无完全相同的则匹配指定编辑距离阀值的结果.
        3. 若还是没有, 返回空列表.

        Args:
            keyWord: (Python String) 输入的keyWord
        
        Return:
            Python 列表, 列表中为可能的关键词. 
            根据上述的三种情况返回例子:
                1. [“严格匹配到的关键词”]
                2. ["可能的关键词1", "可能的关键词2", ... ]
                3. [] (空列表)
        """
        
        keyWordList = []
        for i in self.corpora:
            if strictMatch(keyWord, i):
                return [i]
            elif levenshteinDistance(keyWord, i) <= threshold:
                keyWordList.append(i)
            else:
                pass
        else:
            return keyWordList


