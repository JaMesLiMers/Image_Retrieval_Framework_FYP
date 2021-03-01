from json.decoder import JSONDecodeError
import os
import sys
import cgi
import json
from http.server import BaseHTTPRequestHandler

# solve path problem
abs_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(abs_path) + os.path.sep + ".")
project_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
sys.path.append(project_path)

from Models.Bm25LMIR.archLmirBm25Model import archLmirBm25Model

class PostHandler(BaseHTTPRequestHandler):
    """
    Http服务的handler类, 实现了post请求的处理.

        要求post请求的'Content-type'为'application/json'而且具有我
        们下面定义的输入内容, 否则返回400.
            可识别的输入内容示例:
                {
                    "query": ["图书馆", "苏州"],
                    "weight": [0.1, 0.9]
                }
            返回的json文件格式如下:
                {
                    "result":{
                        "imageId":[图片id列表],
                        "imageSim": [图片的相似度, 归一化到0-1区间内]
                        "imagePath":[图片path列表],
                        "imageTitle":[图片title列表],
                        "imageAnno": [要显示的图片标注列表],
                        }
                    "status":{
                        "statusCode": 0 or 1,
                        "statusMsg": "String"
                    }
                }

        TODO: 
        1. 添加注释内容. [Done]
        2. 增加有weight和无weight的不同行为, 目前默认使用weight. [Done]
        3. 增加服务器启动的参数, 包括ip地址和port. [Done]
        4. 测试可用性. [Done]
        5. 增加状态码. [Done]
    """
    # 在实例化之前加载的全局变量(查询类)
    archPath = "./Dataset/Arch/DemoData_20201228.json"
    model = archLmirBm25Model(archPath=archPath)

    def _information_retrieval(self, query, weight, limit=0):
        """
        使用weight(或者不用)和query进行信息检索的方法.

        Input:
            query: 处理好的字符串列表.
            weight: 处理好的权重列表, 可以留空.
        
        Return:
            返回格式如下的python字典:
                {
                    "result":{
                        "imageId":[图片id列表],
                        "imageSim": [图片的相似度, 归一化到0-1区间内]
                        "imagePath":[图片path列表],
                        "imageTitle":[图片title列表],
                        "imageAnno": [要显示的图片标注列表],
                        }
                    "status":{
                        "statusCode": 0 or 1,
                        "statusMsg": "String"
                    }
                }
            若结果出了问题则返回空字典
        """
        # check weight is same length as query
        useWeight = weight and len(weight) == len(query)

        # init result
        result = {}
        result["result"] = {}
        result["status"] = {}
        imagePaths = []
        imageTitles = []
        imageAnno = []

        # search a list of word/sentence
        try:
            if useWeight:
                sortedResult, index, copora, annoIds, imageIds = self.model.searchWords(listWords=query, weights=weight)
            else:
                sortedResult, index, copora, annoIds, imageIds = self.model.searchSentence(listWords=query)
        except Exception as e:
            result["status"]["statusCode"] = 1
            result["status"]["statusMsg"] = "Fail, catch exception: {} when retrieving".format(e)
            return result

        # limit number of the result entry's 
        if limit != 0 and limit < len(sortedResult):
            sortedResult = sortedResult[0:limit]
            index = index[0:limit]
            copora = copora[0:limit]
            annoIds = annoIds[0:limit]
            imageIds = imageIds[0:limit]


        # get all result
        for (annoId, imageId) in zip(annoIds, imageIds):
            imagePaths.append(self.model.archDataset.imgs[imageId]["targetUrl"])
            imageTitles.append(self.model.archDataset.anns[annoId]["title"])

            # init string to show
            annoString = ""
            annoDict = self.model.archDataset.anns[annoId]
            for key, value in annoDict.items():
                if key == "concateText" or key == "cutConcateText" or key == "labels":
                    continue
                else:
                    annoString += "{} : {}\n".format(key, value)

            imageAnno.append(annoString)
        else:
            # result
            result["result"]["imageId"] = imageIds
            result["result"]["imagePath"] = imagePaths
            result["result"]["imageTitle"] = imageTitles
            result["result"]["imageAnno"] = imageAnno
            result["result"]["imageSim"] = sortedResult.tolist()
            # statue
            result["status"]["statusCode"] = 0
            result["status"]["statusMsg"] = "Success"
        return result

    def _set_headers(self):
        """设定返回的头部"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_HEAD(self):
        """返回头部"""
        self._set_headers()

    # POST echoes the message adding a JSON field
    def do_POST(self):
        """
        处理post请求, 保证输入为json格式并具有目标格式的内容, 并返回需要的结果(json格式)
        """
        # ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
        
        # removed
        # refuse to receive non-json content
        # if ctype != 'application/json':
        #     self.send_response(400)
        #     self.end_headers()
        #     return
            
        # read the message and convert it into a python dictionary
        try:
            length = int(self.headers.get('content-length'))
            message = json.loads(self.rfile.read(length))
        except JSONDecodeError as e:
            self.send_response(400)
            print("content type is not json (cannot decode)\n")
            self.end_headers()

        
        # get input query and weight, if not exist, refuse.
        try:
            query = message['query']
            weight = message['weight']
        except KeyError as e:
            self.send_response(400)
            print("Didnt find 'query' and 'weight' key.")
            self.end_headers()
            return
        
        # get result entry limit
        # default limit to 5000 result entry
        try:
            limit = message['limit']
            limit = int(limit)
        except Exception as e:
            limit = 0

        # generate result
        result = self._information_retrieval(query, weight, limit=limit)

        # send the message back
        self._set_headers()
        self.wfile.write(json.dumps(result).encode("utf-8"))
        return

if __name__ == '__main__':
    """启动server代码"""
    import argparse
    from http.server import HTTPServer
    # add parser, now we can set ip and port.
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", default="0.0.0.0", help="ip address of server", required=False)
    parser.add_argument("-p", "--port", type=int, default=35008, help="port of server", required=False)
    args = parser.parse_args()
    # run server
    server = HTTPServer((args.ip, args.port), PostHandler)
    print('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()

    # test 
    # curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"query":["a", "b"], "weight":[0.1, 0.9]}' -o testresult.json http://127.0.0.1:35008