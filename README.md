# Image_Retrieval_Framework_FYP

## Environment setup

- Clone the repository 
```
git clone https://github.com/JaMesLiMers/Image_Retrieval_Framework_FYP.git
```

- Setup python environment
```
conda create -n FYP python=3.8
conda activate FYP
pip install -r requirement.txt
```

## Dataset prepare
- Sogou Chinese news dataset:["SogouCA"](http://www.sogou.com/labs/resource/ca.php) 
<br/>Since downloading dataset need registration so no link is provided.
<br/>Just put the downloaded `.zip` file into `./Dataset/Sogou/`.

- unzip it then delete original `.zip` file: 
```
unzip ./Dataset/Sogou/news_tensite_xml.full.zip -d ./Dataset/Sogou/
rm ./Dataset/Sogou/news_tensite_xml.full.zip
```

- Extract the corpus we need:
```
cat ./Dataset/Sogou/news_tensite_xml.dat | iconv -f gbk -t utf-8 -c | grep "<content>" > ./Dataset/Sogou/corpus.txt
```

- The corpus format will like this:
```
<content>WTM一点错也没有呢!</content>
```

- Tokenization the chinese content with jieba (break sentence into words):
<br/>(You can also use the different stop words by targeting `-s` flag.)
```
Python ./Preprocessing/Text/Sogou/tokenization.py -i ./Dataset/Sogou/corpus.txt -o ./Dataset/Sogou/corpusSegDone.txt
```

- Train word2vec:
```
Python ./Preprocessing/Text/Sogou/word2vec.py -d ./Dataset/Sogou/corpusSegDone.txt -s ./Dataset/Sogou/word2vecDone.model
```

- Test word2vec result by using T-SNE and PCA:
```
Python ./Preprocessing/Text/Sogou/visualization.py -s ./Dataset/Sogou/word2vecDone.model
```

# TODO: 
- 写LMIR和BM25的使用方法
参考: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.94.8019&rep=rep1&type=pdf




