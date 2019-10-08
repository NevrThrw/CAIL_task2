import warnings
import re
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import jieba


data_input = {1: "divorce/data.txt", 2: "labor/data.txt", 3: "loan/data.txt"}
doc2vec_model_path = {1: "divorce/divorce.model", 2: "labor/labor.model", 3: "loan/loan.model"}
punction = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."


def line_processing(line):  # 提取每行数据的文本内容
    line = line.strip().split('\t')
    sentence = line[1]
    sentence = re.sub(r'[{}]'.format(punction), ' ', sentence).split(' ')
    sent = []
    for sub_sentence in sentence:
        sent.extend(list(jieba.cut(sub_sentence)))
    return line[0], sent, line[2]


def corpusConstruct(data_path):  # 构建Doc2Vec训练集
    Corpus = []
    data = open(data_path, 'r', encoding='utf-8')
    for line in data.read().splitlines():
        tag, content, _ = line_processing(line)
        Corpus.append(TaggedDocument(content, [int(tag)]))
    return Corpus


def train(data_path, model_path):  # 训练模型
    train_text = corpusConstruct(data_path)
    model = Doc2Vec(vector_size=64, min_count=2, epochs=40)
    model.build_vocab(train_text)
    model.train(train_text, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(model_path)


def main():
    for i in range(1, 4):
        train(data_input[i], doc2vec_model_path[i])


if __name__ == '__main':
    main()
