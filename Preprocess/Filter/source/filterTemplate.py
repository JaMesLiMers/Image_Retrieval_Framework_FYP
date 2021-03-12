import re

def replaceNoneCharactor(rawText, toReplace=" "):
    """替换中文英文以外的字符为指定字符
    
    例子: “中文《》English” -> "中文  English"
    
    Args:
        rawText: 有或者没有乱七八糟东西的字符串
        toReplace: 用来替换侦测到对应格式的字符串的目标字符串(默认为空格)
    Return: 
        替换后的字符串
    """
    # filter out all charactor
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z]")
    # replace to space
    cleanText = re.sub(cop, toReplace, rawText)
    return cleanText


def replaceHTML(rawText, toReplace=" "):
    """替换HTML标签为指定字符

    例子: "<p></p>" -> " "

    Args:
        rawText: 包含/不包含html标签的的字符串
        toReplace: 用来替换侦测到对应格式的字符串的目标字符串(默认为空格)
    Return: 
        替换后的字符串
    """
    cop = re.compile('<.*?>')
    cleanText = re.sub(cop, toReplace, rawText)
    return cleanText

def replaceURL(rawText, toReplace=" "):
    """替换URL为指定字符

    例子: "http://baidu.com" -> " "

    Args:
        rawText: 包含/不包含URL的字符串
        toReplace: 用来替换侦测到对应格式的字符串的目标字符串(默认为空格)
    Return: 
        替换后的字符串
    """
    cop = re.compile('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]')
    cleanText = re.sub(cop, toReplace, rawText)
    return cleanText

def replaceJsonKey(rawText, toReplace=" "):
    """替换json中key的部分为指定字符

    例子: '{"sampleKey": sampleData}' -> '{ sampleData}'

    Args:
        rawText: 包含/不包含html标签的的字符串
        toReplace: 用来替换侦测到对应格式的字符串的目标字符串(默认为空格)
    Return: 
        替换后的字符串
    """
    cop = re.compile('"[\s\S]*?" ?:')
    cleanText = re.sub(cop, toReplace, rawText)
    return cleanText

def replaceMultiSpace(rawText, toReplace=" "):
    """替换空格为指定字符

    例子: "a  b     c" -> "a b c"

    Args:
        rawText: 包含/不包含多个空格的字符串
        toReplace: 用来替换侦测到对应格式的字符串的目标字符串(默认为空格)
    Return: 
        替换后的字符串
    """
    cop = re.compile("\s*")
    cleanText = re.sub(cop, toReplace, rawText)
    return cleanText



# Test
if __name__ == "__main__":

    # 替换HTML标签为指定字符
    rawText = "<p></p>"
    print(replaceHTML(rawText))

    # 替换json中key的部分为指定字符
    rawText = '{"sampleKey": sampleData}'
    print(replaceJsonKey(rawText))

    # 替换中文英文以外的字符为指定字符
    rawText = "中文《》::“()&*%…………¥%#¥#¥@#English"
    print(replaceNoneCharactor(rawText))

    # 替换URL为指定字符
    rawText = "http://baidu.com"
    print(replaceURL(rawText))

    # 替换多个空格为指定字符
    rawText = "a  b     c"
    print(replaceMultiSpace(rawText))
