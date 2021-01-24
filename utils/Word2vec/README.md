## Dataset prepare for word2vec mdoel
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
Python ./utils/Word2vec/tokenization.py -i ./Dataset/Sogou/corpus.txt -o ./Dataset/Sogou/corpusSegDone.txt
```

- Train word2vec:
```
Python ./utils/Word2vec/word2vec.py -d ./Dataset/Sogou/corpusSegDone.txt -s ./Models/Doc2Vec/source/trained_model/Sogouword2vecDone.model
```
The result will stored in the `"./Models/Doc2Vec/source/trained_model"` folder.

- Test word2vec result by using T-SNE and PCA:
```
Python ./utils/Word2vec/visualization.py -s ./Models/Doc2Vec/source/trained_model/word2vecDone.model
```