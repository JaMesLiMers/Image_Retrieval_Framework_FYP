from Model.Text.freq_feature import cal_corpus_tf, cal_corpus_tp, cal_all_corpus_tf, cal_idf_BM25, BM25

def cal_all_feature_for_BM25(corpora):
    """计算BM25任务需要的各种特征并返回

    LIMR任务需要的特征包括:

            N: 语料库的总长度, 为int.              (BM25)
            all_idf: 所有语料中词的idf             (BM25)
            corpus_tf: 单个语料中的词的词频         (BM25)
            corpus_length: 语料库各个句子的长度     (BM25)
            avg_doc_length: 语料库的平均文档长度    (BM25)

    这个函数负责计算这些值.

    Args:
        corpora: 多个语料组成的列表, 应该是一个嵌套Python列表. For example:

            [["There", "is", "a", "cat."],
            ["There", "is", "a", "dog."],
            ["There", "is", "a", "wolf."]]

    Return:
        函数有四个返回值, 分别对应:

            N: 语料库的总长度, 为int.
            all_idf: 所有语料中词的idf, 是一个Python字典, 和上文相似, 总结了所有的语料中的词的出现频率特征.
            corpus_tf: 单个语料中的词的词频, 列表中每一个元素都是Python字典, 按位置对应每一条语料的词频特征.
            corpus_length: 语料库各个句子的长度, 为Python List, 每一位置对应一个句子的长度.
            avg_doc_length: 语料库的平均文档长度, 公式为: 文本长度总量/语料库总长度
    """
    N = len(corpora)
    all_tf = {}
    all_idf = {}
    corpus_tf = []
    corpus_length = []

    # cal length, term frequence for every corpus
    for i in corpora:
        tf = cal_corpus_tf(i)
        corpus_length.append(len(i))
        corpus_tf.append(tf)

    # cal average doc length
    avg_doc_length = sum(corpus_length) / N

    # cal all term prob
    all_tf = cal_all_corpus_tf(corpus_tf, sentence_wide=True)

    all_idf = cal_idf_BM25(N, all_tf)

    return N, all_idf, corpus_tf, corpus_length, avg_doc_length

if __name__ == "__main__":
    # Corpora prepare
    corpus = [
        "Hello there good man!",
        "It is quite windy in London",
        "How is the weather today?"
    ]

    tokenized_corpus = [doc.split(" ") for doc in corpus]

    N, \
    all_idf, \
    corpus_tf, \
    corpus_length, \
    avg_doc_length = cal_all_feature_for_BM25(tokenized_corpus)

    # query prepare
    query = "windy London"
    tokenized_query = query.split(" ")

    # [0, 0.9372947225064051, 0]
    print(BM25(tokenized_query, 
               N, 
               all_idf, 
               corpus_tf, 
               corpus_length, 
               avg_doc_length, 
               k1=1.5, 
               b=0.75))
               