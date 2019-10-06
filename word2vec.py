import sklearn
import numpy as np
import multiprocessing
import gensim.models.word2vec
import jieba

data_input = {1: "divorce/data.txt", 2: "labor/data.txt", 3: "loan/data.txt"}
jieba_data_output = {1: "divorce/jieba_data.txt", 2: "labor/jieba_data.txt", 3: "loan/jieba_data.txt"}
word_dict_file = 'word_dict.dict'


def line_processing(line):  # 提取每行数据的文本内容
    line = line.strip().split('\t')
    sentence = line[1]
    sentence = list(jieba.cut(sentence))  # 分词
    return line[0], sentence, line[2]


def line_jieba_processing(line):  # 提取分词数据集的文本内容
    line = line.strip().split('\t')
    sentence = line[1].split()
    return line[0], sentence, line[2]


def construct_word_dict():  # 构建常见词字典和分词后的数据集
    word_dict = dict()
    text=[]
    for i in range(1, 4):
        f = open(data_input[i], 'r')
        j = open(jieba_data_output[i], 'w')
        lines = f.read().splitlines()
        for line in lines:
            no, sentence, vec = line_processing(line)
            text.append(sentence)
            j.write(no + '\t' + ' '.join(sentence) + '\t' + vec + '\n')
            for word in sentence:
                word_dict[word] += 1
        f.close()
        j.close()
    text = [[word for word in senten if word_dict[word]>1] for senten in text]
    Dict = gensim.corpora.Dictionary(text)
    Dict.save(word_dict_file)

if __name__ == '__main__':
    pass
