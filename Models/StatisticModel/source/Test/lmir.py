from Model.Text.freq_feature import cal_corpus_tf, cal_corpus_tp, cal_all_corpus_tf,\
                                    jelinek_mercer, dirichlet, absolute_discount

def cal_all_feature_for_LIMR(corpora):
    """计算LIMR任务需要的各种特征并返回

    LIMR任务需要的特征包括:

            N: 语料库的总长度, 为int.              (LMIR)
            all_tp: 所有语料中词的出现概率          (LMIR)
            corpus_tf: 单个语料中的词的词频         (LMIR)
            corpus_tp: 单个语料中的词的出现概率      (LMIR)
            corpus_length: 语料库各个句子的长度     (LMIR)

    这个函数负责计算这些值, 对于单个语料返回列表, 对于所有语料返回字典.

    Args:
        corpora: 多个语料组成的列表, 应该是一个嵌套Python列表. For example:

            [["There", "is", "a", "cat."],
            ["There", "is", "a", "dog."],
            ["There", "is", "a", "wolf."]]

    Return:
        函数有四个返回值, 分别对应:
            N: 语料库的总长度, 为int.
            all_tp: 所有语料中词的出现概率, 是一个Python字典, 总结了所有的语料中的词的出现频率特征.
            corpus_tf: 单个语料中的词的词频, 列表中每一个元素都是Python字典, 按位置对应每一条语料的词频特征.
            corpus_tp: 单个语料中的词的出现概率, 列表中每一个元素都是Python字典, 按位置对应每一条语料中的词的出现频率特征.
            corpus_length: 语料库各个句子的长度, 为Python List, 每一位置对应一个句子的长度.
    """
    N = len(corpora)
    all_tp = []
    corpus_tf = []
    corpus_tp = []
    corpus_length = []


    # cal length, term frequence, term prob for every corpus
    for i in corpora:
        tf = cal_corpus_tf(i)
        corpus_length.append(len(i))
        corpus_tf.append(tf)
        corpus_tp.append(cal_corpus_tp(tf))

    # cal all term prob
    all_tf = cal_all_corpus_tf(corpus_tf)
    all_tp = cal_corpus_tp(all_tf)

    return N, all_tp, corpus_tf, corpus_tp, corpus_length


if __name__ == "__main__":
    # Corpora prepare
    corpora = [ ["there", "is", "a", "cat"],
                ["there", "is", "a", "dog"],
                ["there", "is", "a", "wolf"]]

    N, \
    all_tp, \
    corpus_tf, \
    corpus_tp, \
    corpus_length = cal_all_feature_for_LIMR(corpora)

    print("N: {}".format(N))
    print("corpus_length: {}".format(corpus_length))
    print("corpus_tf: {}".format(corpus_tf))
    print("corpus_tp: {}".format(corpus_tp))
    print("all_tp: {}".format(all_tp))

    query = ["is", "there", "a", "cat"]

    # Test LMIR.JM
    result_jm = jelinek_mercer(query, 
                               N, 
                               all_tp, 
                               corpus_tp)
    print(result_jm)

    # Test LMIR.DIR
    result_dir = dirichlet(query, 
                           N, 
                           all_tp, 
                           corpus_tf, 
                           corpus_length)
    print(result_dir)

    # Test LMIR.ABS
    result_abs = absolute_discount(query, 
                                   N, 
                                   all_tp, 
                                   corpus_tf, 
                                   corpus_length)
    print(result_abs)
