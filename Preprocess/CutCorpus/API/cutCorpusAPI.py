import Preprocess.CutCorpus.source.cutCorpus as cutSource

def cutChinese(rawText):
    """中文分词

    首先试图将句子最精确地切开, 随后对长词再次切分.

    例子: "小明硕士毕业于中国科学院计算所" -> [小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所]
    
    Args:
        rawText: 需要分词的基本字符串
    Return:
        返回包含Python字符串的List
    """
    return cutSource.cutByJieba(rawText)