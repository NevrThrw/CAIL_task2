import multiprocessing
import json
import BaiduTrans
import time

tags_dict = {1: "divorce/tags.txt", 2: "labor/tags.txt", 3: "loan/tags.txt"}
file_dict = {1: "divorce/data_small_selected.json", 2: "labor/data_small_selected.json",
             3: "loan/data_small_selected.json"}
output_dict = {1: "divorce/data.txt", 2: "labor/data.txt", 3: "loan/data.txt"}


def extract_tags_dict(tags_file):
    # 构建标签字典
    tags = {}
    for tag in tags_file.readlines():
        tags[tag.strip()] = len(tags)
    return tags


def extract_train_data(tags_file, data_file, output_file, tag):
    tags = extract_tags_dict(tags_file)
    j = 1
    for line in data_file.readlines():
        data = json.loads(line, encoding='utf-8')
        for dic in data:
            print(tag, j)
            label = [0] * len(tags)
            sentence = dic['sentence'].strip()
            # time.sleep(BaiduTrans.TIME_SLOT)  # 控制百度翻译API调用频率
            # sentence2 = BaiduTrans.translate(sentence, 'zh', 'en')
            # time.sleep(BaiduTrans.TIME_SLOT)  # 控制百度翻译API调用频率
            # sentence2 = BaiduTrans.translate(sentence2, 'en', 'zh')
            for l in dic['labels']:
                label[tags[l]] = 1
            output_file.write(str(j) + '\t' + sentence + '\t' + " ".join(list(map(str, label))) + '\n')
            j += 1
            # output_file.write(str(j) + '\t' + sentence2 + '\t' + " ".join(list(map(str, label))) + '\n')
            # j += 1
    output_file.close()
    data_file.close()
    tags_file.close()


if __name__ == "__main__":
    pool = []
    for i in range(1, 4):
        data_input_file = open(file_dict[i], encoding='utf-8')
        tags_input_file = open(tags_dict[i], encoding='utf-8')
        data_output_file = open(output_dict[i], 'w', encoding='utf-8')
        process = multiprocessing.Process(target=extract_train_data, args=(
        tags_input_file, data_input_file, data_output_file, tags_dict[i].split('/')[0]))
        pool.append(process)
        process.start()
    for process in pool:
        process.join()
        # extract_train_data(tags_input_file,data_input_file,data_output_file,tags_dict[i].split('/')[0])