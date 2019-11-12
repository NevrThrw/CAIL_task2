import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVC
from gensim.models.doc2vec import Doc2Vec
import re
import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib 
import random
import json
import data_loader as dl
import data_writer as dw

divorce_model=Doc2Vec.load("../divorce/divorce.model") #divorce的Doc2Vec模型
loan_model=Doc2Vec.load("../loan/loan.model")#loan的Doc2Vec模型
labor_model=Doc2Vec.load("../labor/labor.model")#labor的Doc2Vec模型

test_ratio = 0.3
punction = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."


def splitDataSet(X,Y,modify=False,ratio=test_ratio): #构建训练集和测试集
    if modify: #调整正负样本比例
        X_true=[]
        X_false=[]
        Y_true=[]
        Y_false=[]
        for i in range(len(Y)):
            if Y[i]==1:
                X_true.append(X[i])
                Y_true.append(Y[i])
            else:
                X_false.append(X[i])
                Y_false.append(Y[i])
        true_num = len(X_true)
        false_num = true_num*ratio
        for i in range(0,len(X_false),int(len(X_false)/false_num)):
            X_true.append(X_false[i])
            Y_true.append(Y_false[i])
        X_train,X_test,Y_train,Y_test =  train_test_split(X_true,Y_true,test_size=test_ratio,shuffle=True)
    else:
        X_train,X_test,Y_train,Y_test =  train_test_split(X,Y,test_size=test_ratio,shuffle=True)
    return X_train,X_test,Y_train,Y_test


def transferTagVec(tag_dict,tags):
    tagVec=[]
    for key,val in tag_dict.items():
        if key in tags:
            tagVec.append(1)
        else:
            tagVec.append(0)
    return tagVec

def reverseTagVec(tag_dict,tagVec):
    tags=[]
    for i in range(len(tag_dict.keys())):
        if tagVec[i]==1:
            tags.append(tag_dict.keys()[i])
    return tags

def trainSVM(d2vModel,model_path,tag_path,input_path):
    print("Initializing data loader")
    loader = dl.DataLoader(input_path,tag_path)
    label_size = loader.tag_cnt
    X=[]
    Y=[]
    print("Loading data into data loader")
    for content in loader.data:
        for sub_con in content:
            text = sub_con['sentence']
            tags = sub_con['labels']
            text = re.sub(r'[{}]'.format(punction),' ',text).split(' ')
            text = [jieba.cut(i) for i in text if i!='']
            X.append(d2vModel.infer_vector(text))
            Y.append(transferTagVec(tags))
    X = np.array(X)
    Y = np.array(Y)
    Y = Y.transpose()
    print("Beginning trainning SVC")
    for i in range(1,label_size+1):
        classifier = SVC(gamma='auto')
        print("SVC {} training finished".format(i))
        classifier.fit(X,Y[i-1])
        joblib.dump(classifier,model_path+str(i)+'.model')
    print("Train process end")
        
class BRPredictor:
    def __init__(self,svcmodel_path,d2vmodel_path,tag_path):
        self.models=[]
        with open(tag_path,'r',encoding='utf-8') as f: # read data tag file if tag_path passed
                for line in f:
                    self.tag_cnt+=1
                    self.tag_dict[line.strip()]=self.tag_cnt
                    
        for i in range(1,self.tag_cnt+1):
            path = svcmodel_path+str(i)+'.model'
            self.models.append(joblib.load(path))
        
        self.d2v = Doc2Vec.load(d2vmodel_path)
        
    def content_process(self,content):
        text = re.sub(r'[{}]'.format(punction),' ',content).split(' ')
        text = [jieba.cut(i) for i in text if i!='']
        return text
    
    def predict(self,content):
        content = content_process(content)
        x = self.d2v.infer_vector(content)
        tagVec=[]
        for model in self.models:
            tagVec.append(model.predict([x]))
        return reverseTagVec(self.tag_dict,tagVec)
    
    def predictData(self,data_writer,data_loader):
        for content in data_loader.data:
            data_line=[]
            for sub_content in content:
                sub_content['labels']= self.predict(sub_content['sentence'])
                data_line.append(sub_content)
            data_writer.writeContent(data_line)
        data_writer.writeJson()

        
if __name__=="__main__":
    # ------------labor--------------------
    print("-----------------labor------------------")
    trainSVM(labor_model,"../labor/","../labor/tags.txt","../labor/data_small_selected.json")
    predictor = BRPredictor("../labor/","../labor/labor.model","../labor/tags.txt")
    writer = dw.DataWriter("../output/labor/output.json")
    loader = dl.DataLoader("../labor/data_small_selected.json")
    predictor.predictData(writer,loader)



    # ------------loan---------------------
    print("-----------------loan------------------")
    trainSVM(labor_model,"../loan/","../loan/tags.txt","../loan/data_small_selected.json")
    predictor = BRPredictor("../loan/","../loan/loan.model","../loan/tags.txt")
    writer = dw.DataWriter("../output/loan/output.json")
    loader = dl.DataLoader("../loan/data_small_selected.json")
    predictor.predictData(writer,loader)



    # ------------divorce-------------------
    print("-----------------divorce------------------")
    trainSVM(labor_model,"../divorce/","../divorce/tags.txt","../divorce/data_small_selected.json")
    predictor = BRPredictor("../divorce/","../divorce/divorce.model","../divorce/tags.txt")
    writer = dw.DataWriter("../output/divorce/output.json")
    loader = dl.DataLoader("../divorce/data_small_selected.json")
    predictor.predictData(writer,loader)