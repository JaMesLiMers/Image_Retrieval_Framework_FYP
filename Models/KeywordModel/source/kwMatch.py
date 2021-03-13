import editdistance

def levenshteinDistance(keyWord, targetWord):
    """计算两个词语之间的编辑距离
    
    计算编辑距离算法的快速版本
    论文: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.19.7158&rep=rep1&type=pdf

    Args:
        keyWord: (Python String) 输入的keyWord
        targetWord: (Python String) 需要匹配的Word

    Return:
        返回两个词语之间的编辑距离
    
    """
    return editdistance.distance(keyWord, targetWord)

def strictMatch(keyWord, targetWord):
    """严格衡量两个词语是否一样

    Args:
        keyWord: (Python String) 输入的keyWord
        targetWord: (Python String) 需要匹配的Word
    
    Return:
        若两个词语相同则返回True, 否则返回False
    """
    return keyWord == targetWord
