import json

class DataWriter:
    def __init__(self,out_path):
        self.out_content=[]
        self.out_path=out_path
        
    def writeJson(self):
        with open(self.out_path,'w',encoding='utf-8') as f:
            for data in self.out_content:
                json.dump(data,f,ensure_ascii=False)
                f.write("\n")
            f.close()
                
    def writeContent(self,content):
        self.out_content.append(content)