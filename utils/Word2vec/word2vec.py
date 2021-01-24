import logging
import os.path
import sys
import multiprocessing

from gensim.models import word2vec
import pandas as pd
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data", default="./Dataset/Sogou/corpusSegDone.txt", type=str, help="tokenized data file directory")
parser.add_argument("-s", "--save", default="./Dataset/Sogou/word2vecDone.model", type=str, help="output save file directory")
parser.add_argument("-S", "--size", default=300, type=int, help="vector size (default is 300)")
args = parser.parse_args()

dataPath = args.data
savePath = args.save
trainSize = args.size

# logging result
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

# Read in the txt file ...
print("\n Set LineSentence for the txt file ...")
trainSentences = word2vec.LineSentence(dataPath)

# training word2vec model
print("\nTraining word2vec model, it may take a while to finish ... ")
model = word2vec.Word2Vec(trainSentences, size=trainSize, workers=multiprocessing.cpu_count())

# get KeyedVectors model (fixed model)
print("Train finished, taking KeyedVectors model ...")
modelKV = model.wv

# save model
print("\nSaving model ...")
model.save(savePath)
modelKV.save(savePath+".kv")
print("\nAll Done !!!")