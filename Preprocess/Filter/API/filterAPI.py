import Preprocess.Filter.source.filterTemplate as template

def filterHTML(rawText):
    """过滤掉所有HTML的标签和URL, 只保留文字部分

    Args:
        rawText: (String) 需要过滤的字符串
    Return: 
        替换后的字符串
    """
    resultText = rawText
    resultText = template.replaceHTML(resultText)
    resultText = template.replaceURL(resultText)
    resultText = template.replaceNoneCharactor(resultText)
    resultText = template.replaceMultiSpace(resultText)
    return resultText


def filterJSON(rawText):
    """过滤掉Key的部分, 只保留Value部分

    Args:
        rawText: (String) 需要过滤的字符串
    Return: 
        替换后的字符串
    """
    resultText = rawText
    resultText = template.replaceJsonKey(resultText)
    resultText = template.replaceNoneCharactor(resultText)
    resultText = template.replaceMultiSpace(resultText)
    return resultText


def filterText(rawText):
    """过滤掉到只保留文字部分

    Args:
        rawText: (String) 需要过滤的字符串
    Return: 
        替换后的字符串
    """
    resultText = rawText
    resultText = template.replaceNoneCharactor(resultText)
    resultText = template.replaceMultiSpace(resultText)
    return resultText
