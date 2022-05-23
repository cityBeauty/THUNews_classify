#coding:utf-8
import jieba
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, request, abort, make_response, jsonify
import json

app = Flask(import_name=__name__)

@app.route('/')
@app.route('/api/judge_news', methods=["GET","POST"])
def login():
    # data={"title":"head","news":"看看你的"}
    a_list=[]
    p_list=request.get_json()
    a_list.append(p_list['news'])
    mlt = joblib.load('../saved_model/mlt.pkl')
    tf = joblib.load('../saved_model/tf.pkl')
    t2 = []
    j = 0
    for i in a_list:  # 注意python的for循环用法
        n = jieba.lcut(str(i))  # 精确模式分词，适合做文本分析，装换成string形式
        t2.append(' '.join(n))  # 将每个标题的分词结果用空格连接起来
        j = j + 1
    test_01 = t2
    test_01 = tf.transform(test_01)
    y = mlt.predict(test_01)
    a_list.append(y[0])
    print(a_list)
    t={}
    t['judge']=a_list[-1]
    res_json = json.dumps(t,ensure_ascii=False)

    # 返回类型{"judge":"时政"}
    #return res_json
    #return res_json, 200, {"Content-Type":"application/json"}
    #return jsonify(token=123456, gender=0)
    return res_json

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port='5000')
