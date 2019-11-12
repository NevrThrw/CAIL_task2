import json


class DataLoader:
    def __init__(self,file_path,tag_path=None):
        self.data=[]
        with open(file_path,'r',encoding='utf-8') as f: # read training or testing data
            for line in f.readlines():
                doc = json.loads(line)
                self.data.append(doc)
        self.tag_dict={}
        self.tag_cnt=0;
        if tag_path:
            with open(tag_path,'r',encoding='utf-8') as f: # read data tag file if tag_path passed
                for line in f:
                    self.tag_cnt+=1
                    self.tag_dict[line.strip()]=self.tag_cnt