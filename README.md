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

- Export the pythonPath for pwd: \
    This script will import pythonPath automatically (In linux & MAC OS env) \
    **You may need to run this script every time you init your program.**
```
(MAC & linux) source ./set_path.sh
(Win CMD) set PYTHONPATH=%PYTHONPATH%;$PWD
(Win Powershell)  $Env:PYTHONPATH=$Env:PYTHONPATH+";$PWD" 
```

## Trained Model download
- Download the architecture dataset sample json file. Put the json file in the `"Dataset/Arch/"`.


- Download the trained word2vec model weight. \
    (To training your word2vec from scrach, refer to the document in `"utils/Word2vec"` and `"Models/Doc2Vec/source"`) \
    Baidu: https://pan.baidu.com/s/1Q3ZmM0E7WoGqu46j6QK1kw  Password: dsem 

## How this project organized

### Folder structure & usage:
```
- Dataloader
- Dataset
- Models
- utils
```

#### Dataloader 
- `Dataloader`: Store the `dataloader` class for information retrieval. \
    Dataloader class is a python class that read the raw file into a format in python that we defined for further use. Current we just have dataloader for our archtecture dataset.

#### Dataset
- `Dataset`: Store the `raw dataset files`. \
    Current we have two pre-defined folder, Sogou and Arch. You may want to put the dataset you download here.

#### Models
- `Models`: Contains the `information retrieval model` we implement. \
    Current we have three class of model: 
    1. **Statistic-based retrieval Model**: `BestMatch25` and `LMIR`.\
        We build above two kinds of algorithms into one class. \
        Check `"archLmirBm25Model.py"` for Architecture Dataset information retrieval.
    2. **Feature-based model**: `Doc2Vec` (`Word2Vec`)\
        We also implemented the Word2Vec model for information retrieval. \
        Check `"archDoc2vecModel.py"` for Architecture Dataset information retrieval.
    3. **Mixed model**: `MixModel` \
        In this model, we combined above two model and mix the retrieval result with specified weight. \
        Check `"archMixModel.py"` for Architecture Dataset information retrieval.

#### utils
- `utils`: Contains the stopwords and training process for Word2Vec model.

## How to use the model on Architecture dataset:
You may want to use above three kinds model in your system. We have shown the sample usage code in the end of class defination code. Just refer to the code fragment in the end of `"archLmirBm25Model.py"`, `"archDoc2vecModel.py"` and `"archMixModel.py"` file.