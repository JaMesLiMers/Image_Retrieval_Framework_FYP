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
    # how to use:
    archPath = "./Dataset/Arch/DemoData_20201228.json"
    # init model
    model = archLmirBm25Model(archPath=archPath)

    def _information_retrieval(self, query, weight):
        # search a list of word/sentence
        index, copora, annoIds , imageIds = self.model.searchSentence(listWords=query)

        # init result
        result = {}
        result["result"] = {}
        result["result"]["imageId"] = imageIds
        imagePaths = []
        imageTitles = []
        imageAnno = []

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
            result["result"]["imagePath"] = imagePaths
            result["result"]["imageTitle"] = imageTitles
            result["result"]["imageAnno"] = imageAnno
        return result

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    # POST echoes the message adding a JSON field
    def do_POST(self):
        ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
        
        # refuse to receive non-json content
        if ctype != 'application/json':
            self.send_response(400)
            self.end_headers()
            return
            
        # read the message and convert it into a python dictionary
        length = int(self.headers.get('content-length'))
        message = json.loads(self.rfile.read(length))
        
        # add a property to the object, just to mess with data
        query = message['query']
        weight = message['weight']

        # generate result
        result = self._information_retrieval(query, weight)

        # send the message back
        self._set_headers()
        self.wfile.write(json.dumps(result).encode("utf-8"))
        return

if __name__ == '__main__':
    from http.server import HTTPServer
    server = HTTPServer(('localhost', 8080), PostHandler)
    print('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()

    # test 
    # curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"query":["a", "b"], "weight":[0.1, 0.9]}' http://127.0.0.1:8080