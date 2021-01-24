from gensim.models import Word2Vec, KeyedVectors

"""
So, We have already trained a Word2Vec model. 
Then how to use the Word2Vec and KeyedVectors model?
Read these doc for further kwnoledge:
keyedvectors:
https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors
word2vec:
https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.LineSentence
"""

# you can change this according to your situation
SAVED_MODEL_FILE = "./Dataset/Sogou/word2vecDone.model"
SAVED_KV_FILE = "./Dataset/Sogou/word2vecDone.model.kv"

# First, load the saved model.
word_model = Word2Vec.load(SAVED_MODEL_FILE)
word_vectors = KeyedVectors.load(SAVED_KV_FILE)

# word2Vec model can be continuous training
# word_model.train([["你好", "世界"]], total_examples=1, epochs=1)

# word2vec model can convert to word_vectors (as read only model, can not train any more!)
# word_vectors = word_model.wv

# find most similar word
print("找到最相似的词: ")
result = word_vectors.most_similar(positive=['女性', '服务'], negative=['男性'])
# or use word_vectors.most_similar_cosmul
for i in result:
    print(i)

# find the most doesnt match word.
print("找到最无关的词: ‘早餐 午餐 晚餐 绵羊’")
print(word_vectors.doesnt_match("早餐 午餐 晚餐 绵羊".split()))

# find the similarity of two words.
print("衡量相似度: 女性 - 男性")
similarity = word_vectors.similarity('女性', '男性')
print("result: {}".format(similarity))

# find the most similar word for given word
print("找到与给定词最相似的词: ‘猫’")
result = word_vectors.similar_by_word("猫")
for i in result:
    print(i)

# Compute the Word Mover’s Distance between two documents
sentence_1 = "总统对访问了伊利诺伊箱推州"
sentence_2 = "总统欢迎了棉芽的到来"
print("衡量句子和句子的相似度: ")
print("句子1:  总统对访问了伊利诺伊箱推州")
print("句子2:  总统欢迎了棉芽的到来")
similarity = word_vectors.wmdistance(sentence_1, sentence_2)
print("result: {:.4f}".format(similarity))

# compute the distance between two words
print("衡量词和词的相似度: 媒体 - 媒体")
distance = word_vectors.distance("媒体", "媒体")
print("result: {:.1f}".format(distance))

# compute two set of words similarity
print("衡量两个词的集合的相似度: ['寿司', '店'] - ['日本', '餐厅']")
sim = word_vectors.n_similarity(['寿司', '店'], ['日本', '餐厅'])
print("result: {:.4f}".format(sim))

# get word vector
vector = word_vectors['电脑']




