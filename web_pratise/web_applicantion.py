from flask import Flask, request, abort, make_response, jsonify
#导入simple_server模块
from flask import Flask

app = Flask(import_name=__name__)
@app.route('/test', methods=["GET","POST"])
def home():
    return "home"


#实例化一个服务器设置ip为本机，端口为9527，执行程序为上面的app

app.run(debug=True,host='0.0.0.0',port='9527')
