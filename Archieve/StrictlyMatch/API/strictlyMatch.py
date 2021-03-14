class StrictlyMatchModel:
    def __init__(self, corpora):
        """检索包含关键词的样本

        初始化模型:
            StrictlyMatchModel(corpora)

            Args:
                corpora: 多个语料组成的列表, 应该是一个嵌套Python列表. 
                e.g.
                    [["There", "is", "a", "cat"],
                    ["There", "is", "a", "dog"],
                    ["There", "is", "a", "wolf"]]
        """
        self.corpora = corpora

    def retrieve(self, keyword_list):
        """搜索返回corpora中包含关键词的样本

        Args:
            keyword_list: 关键词列表
            e.g.
                ['library', 'XJTLU', ..., 'modern']

        Return:
            Python dict
                key: 样本, 转为tuple
                value: 含有的关键词的list
                e.g.
                    {['XJTLU', 'is', 'a', 'modern', 'univ']: ['XJTLU', 'modern']}
            若不存在包含关键词的样本呢，返回空dict
        """

        match_corpus = {}
        for sentence in self.corpora:
            for keyword in keyword_list:
                if keyword in sentence:
                    sentence = tuple(sentence)
                    if sentence in match_corpus.keys():
                        match_corpus[sentence].append(keyword)
                    else:
                        match_corpus[sentence] = [keyword]

        return match_corpus


if __name__ == '__main__':
    kw_list = ['现代', '大学', '图书馆']
    corpora = [['两只', '老虎', '爱', '跳舞'],
               ['小兔子', '乖乖', '拔', '萝卜'],
               ['西浦', '是', '一所', '现代', '大学', '图书馆', '藏书', '丰富'],
               ['苏州', '大学', '历史', '悠久']]
    model = StrictlyMatchModel(corpora)
    match_corpus = model.retrieve(kw_list)
    print(match_corpus)
