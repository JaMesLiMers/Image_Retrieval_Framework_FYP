import jieba


def cutByJieba(rawText, returnList=True):
    """使用jieba进行中文分词

    首先试图将句子最精确地切开, 随后对长词再次切分.

    例子: "小明硕士毕业于中国科学院计算所" -> [小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所]
    
    Args:
        rawText: 需要分词的基本字符串
        returnList: 用来替换侦测到对应格式的字符串的目标字符串(默认为True)
    Return:
        若returnList为True则返回Python的List, 若False则返回一个generator.
    """
    return jieba.lcut_for_search(rawText) if returnList else jieba.cut_for_search(rawText)
