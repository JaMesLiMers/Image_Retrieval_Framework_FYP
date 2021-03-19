from Models.StatisticModel.source.lmir_bm25 import BM25_LMIR

if __name__ == "__main__":
    # 对BM25进行测试 (官方文档作为标准)
    # 参考: https://github.com/dorianbrown/rank_bm25
    
    corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
    ]   

    tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25_LMIR(tokenized_corpus)

    query = "windy London"
    tokenized_query = query.split(" ")

    doc_scores = bm25.BM25(tokenized_query)
    # array([0.        , 0.93729472, 0.        ])
    print(doc_scores)

    """ ------------------------------------- """

    # 对LMIR进行测试 (官方文档作为标准)
    # 参考: https://github.com/airalcorn2/LMIR

    doc_1 = "This is document one.".split()
    doc_2 = "This is document two. It contains different words.".split()
    docs = [doc_1, doc_2]

    models = BM25_LMIR(docs)

    print("\nLMIR.JM:")
    print(models.jelinek_mercer("This query has words that are found in the corpus.".split()))
    print(models.jelinek_mercer("No matches.".split()))
    # [1.420195912795572, 2.046651718856845]
    # [0, 0]

    print("\nLMIR.DIR:")
    print(models.dirichlet("This query has words that are found in the corpus.".split()))
    print(models.dirichlet("No matches.".split()))
    # [1.7907619629109297, 1.792755981517794]
    # [0, 0]

    print("\nLMIR.ABS:")
    print(models.absolute_discount("This query has words that are found in the corpus.".split()))
    print(models.absolute_discount("No matches.".split()))
    # [1.6519975268528964, 1.8697210106977669]
    # [0, 0]


