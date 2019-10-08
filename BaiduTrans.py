import http.client
import hashlib
import urllib.parse
import random
import json

TIME_SLOT = 5

def translate(transSentence, fromLan, toLan):
    appid = '20190929000338202'
    secretKey = 'Fozm2bEzIsQ4_1Pwq4cM'
    q=transSentence
    myurl = '/api/trans/vip/translate'
    salt = random.randint(32768, 65536)
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLan+ '&to=' + toLan + '&salt=' + str(
        salt) + '&sign=' + sign
    httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
    httpClient.request('GET',myurl)
    response = httpClient.getresponse()
    jsonResponse = response.read().decode("utf-8")  # 获得返回的结果，结果为json格式
    js = json.loads(jsonResponse)  # 将json格式的结果转换字典结构
    dst = str(js["trans_result"][0]["dst"])  # 取得翻译后的文本结果
    if httpClient:
        httpClient.close()
    return dst


# if __name__=="__main__":
#     zh_en=translate('原告林某某诉称：我与被告经人介绍建立恋爱关系，于1995年在菏泽市民政局办理结婚登记手续。','zh','en')
#     time.sleep(0.8)
#     en_zh = translate('The plaintiff, Lin Mou-mou, complained that I had established a love relationship with the defendant on the recommendation of the defendant. In 1995, I went through the marriage registration formalities in Heze Civil Affairs Bureau.','en','zh')