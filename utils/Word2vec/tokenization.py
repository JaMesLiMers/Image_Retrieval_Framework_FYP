import argparse
import pandas as pd
from tqdm import tqdm
import jieba
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", type=str, help="input file directory", required=True)
parser.add_argument("-o", "--output", type=str, help="output file directory", required=True)
parser.add_argument("-s", "--stopword", default="./Preprocessing/Text/Stop_words/hit_stopwords.txt", type=str, help="stop word's directory")
args = parser.parse_args()

filePath = args.input
# For example './Dataset/Sogou/corpus.txt'
fileSegWordDonePath = args.output
# For example './Dataset/Sogou/corpusSegDone.txt'
stopWordPath = args.stopword

# read in the txt file (only contain content)
print("\nRead in the txt file ...")
# read in stopwords as list
with open(stopWordPath, 'r') as f:
    stopWords = [line.strip() for line in f.readlines()]
fileTrainRead = pd.read_csv(filePath)
fileTrain = pd.Series(fileTrainRead.iloc[:,0])

# delete all <content> ... <content/>
print("\nPreprocessing ...")
f = lambda x: x[9:-11]
fileTrain = fileTrain.apply(f)
print("\nDeleted all <content> ... <content/>.")

# delete the empty line
fileTrain.dropna(how='any')
print("\nDeleted Empty line.")

# cut all by jieba
print("\nCutting words by jieba (with extracting stop words) ...")
fileTrainSeg = []
progress = tqdm(fileTrain)
count = 0

for line in progress:
    data = jieba.cut(line, cut_all=False)
    data = list(data)
    data_ = []
    for word in data:
        if word not in stopWords:
            data_.append(word)

    fileTrainSeg.append(" ".join(list(data_)))

# drop none value
print("\nDroping none value ...")
fileTrainSeg = list(filter(None, fileTrainSeg))

# output
print("\nSaving the new file ...")
output_list = pd.Series(fileTrainSeg)
output_list.to_csv(fileSegWordDonePath, encoding='utf-8', header=False, index=False)
print("\nDone!")