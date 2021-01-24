import sys
import gensim
import jieba

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models.doc2vec import Doc2Vec

from Dataloader.Arch.arch import Arch


class Doc2VecModel:
    def getVecs(self, model, corpus, size):
        """
        把文档向量化
        没用过所以先不瞎写了
        """
        vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]
        return np.concatenate(vecs)

    def train(self, x_train, vector_size=300, epoch_num=5):
        """
        用语料库训练模型
        :param x_train: 语料库
        :param vector_size: 训出来的向量维度
        :param epoch_num: epoch
        :return: 训好的模型
        """
        model_dm = Doc2Vec(x_train, min_count=1, window=5, vector_size=vector_size, sample=1e-3, negative=5, workers=4)
        model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=epoch_num)
        model_dm.save('trained_model/d2v_model.model')

        return model_dm

    def load_model(self, model_path):
        model_dm = Doc2Vec.load(model_path)
        return model_dm

    def test(self, model_dm, test_txt, topn=10):
        """
        相似度排序: 一个文本 -> 语料库所有样本
        :param model_dm: 训好的模型
        :param test_txt: 测试的文本
        :param topn: 返回数量
        :return: topn个最像的样本
        """
        test_text = list(test_txt.split())
        inferred_vector_dm = model_dm.infer_vector(test_text)
        sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=topn)
        return sims

    def retrieve(self, test_text, model_dm, corpus):
        test_vec = np.expand_dims(model_dm.infer_vector(test_text), axis=0)

        sim_array = np.zeros(len(corpus))
        for idx, sample in enumerate(corpus):
            sample_vec = np.expand_dims(model_dm.infer_vector(sample), axis=0)
            sim_array[idx] = cosine_similarity(test_vec, sample_vec)

        return sim_array


if __name__ == '__main__':
    # load archdataset
    ArchDataset = Arch(annotationFile="../../../Dataset/Arch/DemoData_20201228.json", imageFolder=None)
    ArchDataset.reverseCharForAllContext()

    # generate annotation and corpora list
    annIdList = []
    corporaList = []
    corporaList_d2v = []
    TaggededDocument = gensim.models.doc2vec.TaggedDocument  # 方便gensim用的文档对象

    for i, (annotation, content) in enumerate(ArchDataset.anns.items()):
        annIdList.append(annotation)
        corporaList.append(content["cutConcateText"])
        document = TaggededDocument(content["cutConcateText"], tags=[i])
        corporaList_d2v.append(document)

    corpus_len = len(ArchDataset.anns.items())
    d2v_model = Doc2VecModel()

    # 如果没有训好的模型就训一次
    # d2v_trained = d2v_model.train(x_train=corporaList_d2v)
    d2v_trained = d2v_model.load_model('trained_model/d2v_model.model')  # 有的话就load

    test_words = ["平面图", "博物馆"]

    # test_sentence = ' '.join(i for i in test_words)

    sim_arr = d2v_model.retrieve(test_text=test_words, corpus=corporaList, model_dm=d2v_trained)

    print(sim_arr)

    # sim_dict = {}
    # for key_word in test_words:
    #     sim_dict[key_word] = d2v_model.test(model_dm=d2v_trained, test_txt=key_word, topn=corpus_len)
    #     # 对每个词返回了corpora所有样本的 [index, 相似度], 用dict存储
    #
    #     # 打印出来康康
    #     print('\n', key_word)
    #     for i, (count, sim) in enumerate(sim_dict[key_word]):
    #         words = corporaList_d2v[count]
    #         sentence = ' '.join(i for i in words[0])
    #
    #         if i < 5:
    #             print(str(count) + ': ', sentence, sim)

