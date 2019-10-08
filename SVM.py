from sklearn.svm import SVC
from gensim.models.doc2vec import Doc2Vec
import re
import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

divorce_model = Doc2Vec.load("divorce/divorce.model")
loan_model = Doc2Vec.load("loan/loan.model")
labor_model = Doc2Vec.load("labor/labor.model")
data_type = {1: "divorce", 2: "labor", 3: "loan"}
data_input = {1: "divorce/data.txt", 2: "labor/data.txt", 3: "loan/data.txt"}
models = {1: divorce_model, 2: loan_model, 3: labor_model}
label_size = 20
test_ratio = 0.3
punction = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."


def line_processing(line):  # 提取每行数据的文本内容
    line = line.strip().split('\t')
    sentence = line[1]
    sentence = re.sub(r'[{}]'.format(punction), ' ', sentence).split(' ')
    sent = []
    for sub_sentence in sentence:
        sent.extend(list(jieba.cut(sub_sentence)))
    return line[0], sent, line[2]


def constructDataSet(data_path, model_tag):  # 构建X，Y的数据集
    data_file = open(data_path, 'r', encoding='utf-8')
    lines = data_file.read().splitlines()
    X = []
    Y = []
    d2v = models[model_tag]
    for line in lines:
        _, x, y = line_processing(line)
        x = d2v.infer_vector(x)
        X.append(x)
        y = list(map(int, y.split()))
        Y.append(y)
    Y = np.array(Y).transpose()
    return X, Y


def splitDataSet(X, Y):  # 构建训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, shuffle=True)
    return X_train, X_test, Y_train, Y_test


def trainSVM(X, Y, model_path):  # 训练单个分类器
    X_train, X_test, Y_train, Y_test = splitDataSet(X, Y)
    classifier = SVC(gamma='auto')
    classifier.fit(X=X_train, y=Y_train)
    accuracy = classifier.score(X_test, Y_test)
    joblib.dump(classifier, model_path)
    print(model_path, accuracy)


def beginTrain():  # 训练分类器
    for i in range(1, 4):
        X, Y = constructDataSet(data_input[i], i)
        print(len(X), len(Y), len(Y[0]))
        for j in range(len(Y)):
            model_path = data_type[i] + '/label' + str(j + 1) + ".model"
            trainSVM(X, Y[j], model_path)


def predictSingleLabel(x, model_type, label_tag):  # 预测单个标签
    model_path = model_type + '/' + 'label' + str(label_tag + 1) + '.model'
    model = joblib.load(model_path)
    return model.predict(x)


def predict(text, model_type):  # 预测数据的标签
    text = re.sub(r'[{}]'.format(punction), ' ', text).split(' ')
    words = []
    for word in text:
        words.extend(list(jieba.cut(word)))
    d2v = models[model_type]
    x = d2v.infer_vector(words)
    res = []
    for i in range(label_size):
        res.append(predictSingleLabel(x, data_type[model_type], i)[0])
    return res

#
# if __name__ == "__main__":
#     beginTrain()
