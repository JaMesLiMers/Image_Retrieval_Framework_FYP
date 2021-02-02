import sys
import os

# solve path problem
abs_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(abs_path) + os.path.sep + ".")
project_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
sys.path.append(project_path)

from Models.Bm25LMIR.archLmirBm25Model import archLmirBm25Model


"""

Input:
{
    "query": ["图书馆", "苏州"],
    "weight": [0.1, 0.9]
}

Output:
{
    "result":{
        "imageId":[],
        "imagePath":[],
        "imageTitle":[],
        "imageAnno": [],
    },
}

"""


if __name__ == "__main__":
    # how to use:
    archPath = "./Dataset/Arch/DemoData_20201228.json"
    # init model
    model = archLmirBm25Model(archPath=archPath)

    # init inputs:
    inputs = {
        "query": ["图书馆", "苏州"],
        "weight": [0.1, 0.9]}

    # search a list of word/sentence
    index, copora, annoIds , imageIds = model.searchSentence(listWords=["图书馆"])

    # init result
    result = {}
    result["result"] = {}
    result["result"]["imageId"] = imageIds
    imagePaths = []
    imageTitles = []
    imageAnno = []

    # get all result
    for (annoId, imageId) in zip(annoIds, imageIds):
        imagePaths.append(model.archDataset.imgs[imageId]["targetUrl"])
        imageTitles.append(model.archDataset.anns[annoId]["title"])

        # init string to show
        annoString = ""
        annoDict = model.archDataset.anns[annoId]
        for key, value in annoDict.items():
            if key == "concateText" or key == "cutConcateText" or key == "labels":
                continue
            else:
                annoString += "{} : {}\n".format(key, value)

        imageTitles.append(annoString)
    else:
        result["result"]["imagePath"] = imagePaths
        result["result"]["imageTitle"] = imageTitles
        result["result"]["imageAnno"] = imageAnno
    
    print("Test Done")
