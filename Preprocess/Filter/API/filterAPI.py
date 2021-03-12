import Preprocess.Filter.source.filterTemplate as template

def filterHTML(rawText):
    """过滤掉所有HTML的标签和URL, 只保留文字部分

    Args:
        rawText: 有或者没有乱七八糟东西的字符串
        toReplace: 用来替换侦测到对应格式的字符串的目标字符串(默认为空格)
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
    """过滤掉Key的部分, 只保留文字部分

    Args:
        rawText: 有或者没有乱七八糟东西的字符串
        toReplace: 用来替换侦测到对应格式的字符串的目标字符串(默认为空格)
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
        rawText: 有或者没有乱七八糟东西的字符串
        toReplace: 用来替换侦测到对应格式的字符串的目标字符串(默认为空格)
    Return: 
        替换后的字符串
    """
    resultText = rawText
    resultText = template.replaceNoneCharactor(resultText)
    resultText = template.replaceMultiSpace(resultText)
    return resultText
