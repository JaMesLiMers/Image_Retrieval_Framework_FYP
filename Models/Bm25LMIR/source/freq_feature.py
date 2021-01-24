from math import log

# ------------------Statistical feature cal (Preprocessing)---------------------
# heavily Borrowed from these repository:
# https://github.com/airalcorn2/LMIR
# https://github.com/dorianbrown/rank_bm25
# 1. calculate the tf value
# 2. calculate the tp value
# 3. calculate the idf value (according to BM25)

def cal_corpus_tf(corpus):
    """计算语料中各个词出现的频率.

    计算一个语料中每个词出现的频率, 默认一个语料是已经分词完成的形式. (目前空的数据""
    也会被算作是一个词语)

    Args:
        corpus: 一个分词完成的语料, 默认认为这是一个Python列表, 列表中是一个一个分
            好的词. For example:

            ["There", "is", "a", "cat."]

    Returns:
        一个根据输入的语料生成的词频统计表, 是一个Python字典, 字典中的键值对分别是词
        语和在文中的出现次数. For example:

        {'There': 1,
         'is': 1,
         'a': 1,
         'cat': 1,}

        如果输入的语料是空的, 那么返回的词频统计表也应该是一个空的字典.
    """
    token_counts = {}

    # generate counts table
    for token in corpus:
        token_counts[token] = token_counts.get(token, 0) + 1
    return token_counts


def cal_all_corpus_tf(tfs, sentence_wide=False):
    """对输入的所有的词频统计数据进行求和.

    将多个词频统计字典整理在一起, 在一个字典中汇总所有出现过的词语, 如果词频统计的词语重复,
    就将词频统计的值相加. 有两种细粒度级别, 一种是句子级别, 一种是词级别, 默认细粒度是词级别.

    Args:
        tfs: 多个已经生成好的的词频数据的列表, 是一个Python列表, 列表中存的是词频的字典.
            For example:

            [{"cat": 1, "dog": 1}, 
            {"cat": 1, "wolf": 1}, 
            {"dog": 5, "cat": 1},]

        sentence_wide: 控制细粒度是句子级别还是词级别, 若是句子级别, 每个句子中的重复单词将
            只被记录为一次出现. 若是词级别, 每个句子中的重复单词将会被重复计算.

    Return:
        一个汇总了所有的词频数据的总表, 将不同的词语整理在一起, 对相同的词语进行求和, 为一
        个Python的字典. 按照上文的输入例子, 若是词级别(默认), 则返回:

        {"cat": 3,
        "dog": 6,
        "wolf": 1,}

        若是句子级别, 则返回:

        {"cat": 3,
        "dog": 2,
        "wolf": 1,}

        如果输入的词频是空的, 那么返回的统计表也应该是一个空的字典.
    """
    tf_all = {}

    # iterate all and add/merge
    for i in tfs:
        for k,v in i.items():
            if sentence_wide:
                tf_all[k] = tf_all.get(k,0) + 1
            else:
                tf_all[k] = tf_all.get(k,0) + v
    return tf_all


def cal_corpus_tp(tf):
    """计算词频的出现概率.

    计算每个词的出现概率, 输入和返回都应该是一个Python字典. 出现概率的计算方式是:

        特定词的出现概率 = 特定词出现的频率 / 总共的词数

    Args:
        tf: 一个已经生成好的的词频统计表, 默认是一个Python的字典, For example:

            {"cat": 3,
            "dog": 6,
            "wolf": 1,}

    Return:
        返回的是计算好的词的出现概率, 格式和输入的格式相似, For example:
        
        {"cat": 0.3,
        "dog": 0.6,
        "wolf": 0.1,}

    """
    # get total length of doc
    total_length = sum(tf.values())
    
    # calculate p for all token and return p dict
    tp = {
            token: token_count / total_length 
            for (token, token_count) in tf.items()
        }
    return tp

# 关于idf的解析可以参考:
# https://www.cnblogs.com/geeks-reign/p/Okapi_BM25.html

def cal_idf_BM25(N, all_tf, epsilon=0.25):
    """
    Calculates idf in documents and in corpus (BM25 version).

    词语的idf的值可能是负的(如果多半个文档中都包含一个词的化idf值可能是负的), 为了缓解这种
    情况(不能避免)我们将所有的idf值都设定了一个最小值: epsilon * average_idf. (epsilon
    默认为0.25) BM25版本的每个词的idf的计算公式如下:

        log(N - tf + 0.5) - log(tf + 0.5)

    其中N为语料中句子的数量, tf为词在整个语料中出现的次数

    Args:
        all_tf: 所有语料中的词频统计表, 默认是一个python字典. (注意!! 这里使用的是句子级别的
        词频统计, 每一句的多个重复词按照出现一次计算.)
        for example:

            {"cat": 1,
            "dog": 1,
            "wolf": 1,}

        N: 语料库中的语句条目数, 为int.
        epsilon: 参考描述, 用来决定idf的最小值, 默认为0.25.

    Return:
        返回对出现的所有单词的idf值的统计表, 为一个Python字典, for example:

            {'cat': 0.5108256237659907,
             'dog': 0.5108256237659907,
             'wolf': 0.5108256237659907}

    """
    # collect idf sum to calculate an average idf for epsilon value
    idf_sum = 0
    idf = {}
    # collect words with negative idf to set them a special epsilon value.
    # idf can be negative if word is contained in more than half of documents
    negative_idfs = []
    for word, freq in all_tf.items():
        idf_word = log(N - freq + 0.5) - log(freq + 0.5)
        idf[word] = idf_word
        idf_sum += idf_word
        if idf_word < 0:
            negative_idfs.append(word)
    
    # calculate eps and replace negative value
    average_idf = idf_sum / len(idf)
    eps = epsilon * average_idf
    for word in negative_idfs:
        idf[word] = eps
    return idf


# ------------------Statistical feature cal (LMIR)---------------------
# According to this paper:
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.94.8019&rep=rep1&type=pdf
# heavily Borrowed from this repository:
# https://github.com/airalcorn2/LMIR

# LMIR核心算法部分: JM, DIR, ABS
def jelinek_mercer(query_tokens, N, all_tp, corpus_tp, lamb=0.1):
    """Calculate the Jelinek-Mercer scores for a given query.

    计算LMIR.JM值, 计算公式为:

    log((1-lambda)*p_ml + lambda * p)

    其中p_ml是单个文档中对应词出现的概率(注意!!!这里的每个句子中的出现频率是可以大于1的), 
    p是在所有文档中p出现的概率, 为了兼容, 如果对应词没有出现在句子的词频统计中, 则p_ml按
    零来算. 如果对应词没有出现在所有的词频统计中, 则跳过对应的计算(也就是说, 如果字符全都
    没有重复, 分数就是0).

    Args:
        query_tokens: 用来查询的句子, 默认已经进行了分词操作. For example:

            [“Is”, "there", "a", "cat"]

        N: 语料库中的语句条目数, 为int.

        corpus_tp: 用于查询的句子(数据库中处理好的)的词语出现概率表, 是键值对为词和出现概
            率的字典组成的Python列表, 在列表中的位置与句子处理前的相对位置一一对应. For 
            example:

            [{"cat": 0.5, "dog": 0.5}, 
            {"cat": 0.5, "wolf": 0.5}, 
            {"dog": 0.825, "cat": 0.175},]

        all_tp: 用于查询的句子(数据库中处理好的)的词语的总的出现概率表, 和corpus_tp相似但
            是是总的概率表, 所以只是一个Python字典. For example:

            {"cat": 0.3, "dog": 0.6, "wolf": 0.1,}
        
        lamb: 在计算LMIR.JM特征的超参数, 默认值为0.1.

    Return:
        返回对每一个句子的LMIR.JM查询值, 返回一个对应每个句子的分数的列表, N有几个句子的统计
        表就应该有几个查询值. 格式如下:

        [0, 0, 0]

    """
    scores = []
    for doc_idx in range(N):
        p_ml = corpus_tp[doc_idx]
        score = 0
        for token in query_tokens:
            if token not in all_tp:
                continue
        
            p = all_tp[token]

            score -= log((1 - lamb) * p_ml.get(token, 0) + lamb * p)

        scores.append(score)

    return scores

def dirichlet(query_tokens, N, all_tp, corpus_tf, corpus_length, mu=2000):
    """Calculate the Dirichlet scores for a given query.

    计算LMIR.DIR值, 计算公式为:

    log((f(q_i,d) + mu * p(q_i|c)) / (doc_length + mu))

    其中f(q_i, d)为词q_i在单个文本中出现的频率(注意!!!这里的每个句子中的出现频率是可以大于1的),
    p(q_i|c)为词q_i在所有文本中出现的概率.同样, 为了兼容, 如果如果对应词没有出现在句子的词频统计
    中, 则f(q_i, d)按零来算. 如果对应词没有出现在所有词的概率统计中, 则跳过对应的计算(也就是说, 
    如果字符全都没有重复, 分数就是0).

    Args:
        query_tokens: 用来查询的句子, 默认已经进行了分词操作. For example:

                [“Is”, "there", "a", "cat"]

        N: 语料库中的语句条目数, 为int.

        corpus_tf: 用于查询的句子(数据库中处理好的)的词语出现频率表, 是键值对为词和出现频率
                的字典组成的Python列表, 在列表中的位置与句子处理前的相对位置一一对应. 
                For example:

                [{"cat": 0.5, "dog": 0.5}, 
                {"cat": 0.5, "wolf": 0.5}, 
                {"dog": 0.825, "cat": 0.175},]

        all_tp: 用于查询的句子(数据库中处理好的)的词语的总的出现概率表, 和corpus_tp相似但
                是是总的概率表, 所以只是一个Python字典. For example:

                {"cat": 0.3, "dog": 0.6, "wolf": 0.1,}

        corpus_length: 用于标记每一个句子中的词的总数, 默认为一个Python int数组, For example:

                [2, 2, 2,]

        mu: 在计算LMIR.JM特征的超参数, 默认值为2000.

    Return:
        返回对每一个句子的LMIR.DIR查询值, 返回一个对应每个句子的分数的列表, N有几个句子的统计
        表就应该有几个查询值. 格式如下:

        [0, 0, 0]

    """

    scores = []
    for doc_idx in range(N):
        tf = corpus_tf[doc_idx]
        doc_len = corpus_length[doc_idx]
        score = 0
        for token in query_tokens:
            if token not in all_tp:
                continue

            score -= log((tf.get(token, 0) + mu * all_tp[token]) / (doc_len + mu))

        scores.append(score)

    return scores

def absolute_discount(query_tokens, N, all_tp, corpus_tf, corpus_length, delta=0.7):
    """Calculate the absolute discount scores for a given query.

    计算LMIR.ABS值, 计算公式为:

            log(
                max(f(q_i, d) - delta, 0) / doc_len
                + delta * d_uniqe_occer / doc_len * p(p_i|C)
            )
    其中
    
    Args:
        query_tokens: 用来查询的句子, 默认已经进行了分词操作. For example:

                [“Is”, "there", "a", "cat"]

        N: 语料库中的语句条目数, 为int.
        
        corpus_tf: 用于查询的句子(数据库中处理好的)的词语出现频率表(注意!!!这里的每个
                句子中的出现频率是可以大于1的), 是键值对为词和出现频率的字典组成的Python
                列表, 在列表中的位置与句子处理前的相对位置一一对应. 
                For example:

                [{"cat": 0.5, "dog": 0.5}, 
                {"cat": 0.5, "wolf": 0.5}, 
                {"dog": 0.825, "cat": 0.175},]

        all_tp: 用于查询的句子(数据库中处理好的)的词语的总的出现概率表, 和corpus_tp相似但
                是是总的概率表, 所以只是一个Python字典. For example:

                {"cat": 0.3, "dog": 0.6, "wolf": 0.1,}

        corpus_length: 用于标记每一个句子中的词的总数, 默认为一个Python int数组, For example:

                [2, 2, 2,]

        delta: 在计算LMIR.ABS特征的超参数, 默认值为0.7.

    Return:
        返回对每一个句子的LMIR.ABS查询值, 返回一个对应每个句子的分数的列表, N有几个句子的统计
        表就应该有几个查询值. 格式如下:

        [0, 0, 0]
    """

    scores = []
    for doc_idx in range(N):
        tf = corpus_tf[doc_idx]
        doc_len = corpus_length[doc_idx]
        d_u = len(tf)
        score = 0
        for token in query_tokens:
            if token not in all_tp:
                continue

            score -= log(
                max(tf.get(token, 0) - delta, 0) / doc_len
                + delta * d_u / doc_len * all_tp[token]
            )

        scores.append(score)

    return scores


# ------------------Statistical feature cal (BM25)---------------------
# heavily Borrowed from this repository:
# https://github.com/airalcorn2/LMIR
# you can ref this for further read:
# https://www.cnblogs.com/geeks-reign/p/Okapi_BM25.html

# TODO: Rank BM25 写注释
def BM25(query_tokens, N, all_idf, corpus_tf, corpus_length, avg_doc_length, k1=1.5, b=0.75):
    """The ATIRE BM25 variant uses an idf function which uses a log(idf) score. 
    
    BM25算法, 这个算法使用了epsilon来缓解负idf值的影响. 计算公式是:

        score(q, d) = \sum_i \log \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} \cdot \frac{(k_1 + 1)
        \cdot tf(q_i, d)}{k_1(1 - b + b\cdot \frac{L_d}{L_{avg}}) + tf(q_i, d)} \cdot \frac{(k_3 + 1)\cdot tf(q_i, q)}{k_3 + tf(q_i, q)}

    更多的信息请参考 [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine]
    
    Args:
        query_tokens: 用来查询的句子, 默认已经进行了分词操作. For example:

                [“Is”, "there", "a", "cat"]

        N: 语料库中的语句条目数, 为int.

        all_idf: 对出现的所有单词的idf值的统计表, 为一个Python字典, for example:

                {'cat': 0.5108256237659907,
                'dog': 0.5108256237659907,
                'wolf': 0.5108256237659907}
        
        corpus_tf: 用于查询的句子(数据库中处理好的)的词语出现频率表(注意!!!这里的每个
                句子中的出现频率是可以大于1的), 是键值对为词和出现频率的字典组成的Python
                列表, 在列表中的位置与句子处理前的相对位置一一对应. 
                For example:

                {"cat": 1,
                "dog": 1,
                "wolf": 1,}

        corpus_length: 用于标记每一个句子中的词的总数, 默认为一个Python int数组, For example:

                [2, 2, 2,]

        avg_doc_length: 语料库的平均文档长度, 公式为: 文本长度总量/语料库总长度
        k1: 计算时公式参数, 默认为: 1.5
        b: 计算时公式参数, 默认为: 0.75

    Return:
        返回对每一个句子的BM25查询值, 返回一个对应每个句子的分数的列表, N有几个句子的统计
        表就应该有几个查询值. 格式如下:

        [0, 0, 0]

    """
    scores = []
    for doc_idx in range(N):
        score = 0
        doc_len = corpus_length[doc_idx]
        doc_freqs = corpus_tf[doc_idx]
        for token in query_tokens:
            if token not in doc_freqs:
                continue

            q_freq = doc_freqs.get(token, 0)
            score += all_idf.get(token, 0) * (q_freq * (k1 + 1) / 
                                                (q_freq + k1 * (1 - b + b * doc_len / avg_doc_length)))

        scores.append(score)
    return scores


if __name__ == "__main__":
    """测试用代码"""
    pass

    # corpus_1 = ["There", "is", "a", "cat."]
    # corpus_None_1 = [""]
    # corpus_None_2 = []
    # print(cal_corpus_tf(corpus_1))
    # print(cal_corpus_tf(corpus_None_1))
    # print(cal_corpus_tf(corpus_None_2))

    # tfs_1 = [{"cat": 1, "dog": 1}, 
    #         {"cat": 1, "wolf": 1}, 
    #         {"dog": 5, "cat": 1},]
    # tfs_None_1 = [{}]
    # tfs_None_2 = []
    # print(cal_all_corpus_tf(tfs_1))
    # print(cal_all_corpus_tf(tfs_None_1))
    # print(cal_all_corpus_tf(tfs_None_2))
    
    # test_idf = {"cat": 1,
    #             "dog": 1,
    #             "wolf": 1,}
    # print(cal_idf_BM25(test_idf, 3))

    