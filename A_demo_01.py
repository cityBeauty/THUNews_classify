from sklearn.feature_extraction.text import TfidfVectorizer #在文本分类之中，首先分词，然后将分词之后的文本进行tfidf计算，并向量化（这一部分是核心），最后利用传统机器学习算法进行分类就可以了。
from sklearn.naive_bayes import  MultinomialNB#引入朴素贝叶斯
import jieba#jieba库是一款优秀的 Python 第三方中文分词库
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import os
import pandas as pd
import csvkit
import joblib
"""
jieba 支持三种分词模式：精确模式、全模式和搜索引擎模式，下面是三种模式的特点。
精确模式：试图将语句最精确的切分，不存在冗余数据，适合做文本分析
全模式：将语句中所有可能是词的词语都切分出来，速度很快，但是存在冗余数据
搜索引擎模式：在精确模式的基础上，对长词再次进行切分
"""


isir=pd.read_table('D:/QQ下载/qq文件下载/THUCNews/THUCNews/彩票/256824.txt')
# print(isir)
# dict=pd.read_table('data/data12701/dict.txt')
# Test=pd.read_table('data/data12701/Test.txt')
# Test_IDs=pd.read_table('data/data12701/Test_IDs.txt')
# Val_IDs=pd.read_table('data/data12701/Val_IDs.txt')
# Train=pd.read_table('data/data12701/Train.txt')
# Train_IDs=pd.read_table('data/data12701/Train_IDs.txt')
def fenlei():
    # 获取新闻标题数据训练集
    news = pd.read_table('data/cnews/cnews.train.txt',header=None)#默认会自动推断数据文件头,如果设置为None则无文件头,为1则第一行是文件头
    # news = news.drop(1,axis=1) # 去掉中文类别 axis=1指按列
    # print(news)
    text = news[1].tolist()#将数组或者矩阵转换成列表
    # print(text)
    # 对新闻标题列表里面的标题进行分词
    j=0
    t = []
    for i in text:#注意python的for循环用法
        n= jieba.lcut(str(i))	#精确模式分词，适合做文本分析，装换成string形式
        t.append(' '.join(n)) # 将每个标题的分词结果用空格连接起来
        j=j+1
        print(j)
    news[1] = t
    # print(news[1])
    #获取测试集
    test = pd.read_table("data/cnews/cnews.test.txt",header=None)
    text = test[0].tolist()
    t1 = []
    for i in text:
        n = jieba.lcut(str(i))
        t1.append(' '.join(n))  # 将每个标题的分词结果用空格连接起来
    x_test = t1
    label = ["财经","彩票","房产","股票","家具","教育","科技","社会","时尚","时政","体育","星座","游戏","娱乐"]
    # print(x_test)
    # 进行数据分割
    x_train,x_test,y_train,y_test = model_selection.train_test_split(news[1], news[0], test_size=0.2)
    print(x_test)
    print('######')
    print(x_train)
    print(y_train)
    print('######')
    # 对数据集进行特征抽取
    tf = TfidfVectorizer()#TfidfVectorizer可以把原始文本转化为tf-idf的特征矩阵，从而为后续的文本相似度计算，主题模型(如LSI)，文本搜索排序等一系列应用奠定基础。

    # 以训练集当中的词的列表进行每篇文章重要性统计['a','b','c','d',]
    x_train = tf.fit_transform(x_train)

    """
	sklearn里的封装好的各种算法使用前都要fit，fit相对于整个代码而言，为后续API服务。fit之后，然后调用各种API方法，transform只是其中一个API方法，所以当你调用transform之外的方法，也必须要先fit。
	fit_transform(): joins the fit() and transform() method for transformation of dataset.
 	解释：fit_transform是fit和transform的组合，既包括了训练又包含了转换。
	transform()和fit_transform()二者的功能都是对数据进行某种统一处理（比如标准化~N(0,1)，将数据缩放(映射)到某个固定区间，归一化，正则化等）
	fit_transform(trainData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等	（根据具体转换的目的），然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。
	"""
    # print(tf.get_feature_names())
    # print('')
    x_test = tf.transform(x_test)


    # 进行朴素贝叶斯算法的预测
    mlt = MultinomialNB(alpha=0.02)
    print(x_test)
    print('x_test')
    '''
    class sklearn.naive_bayes.MultinomialNB (alpha=1.0,fit_prior=True, class_prior=None)
    其中：
    alpha : 浮点数, 可不填 (默认为1.0)
    拉普拉斯或利德斯通平滑的参数λ \lambdaλ，如果设置为0则表示完全没有平滑选项。但是需要注意的是，平滑相当于人为给概率加上一些噪音，因此λ \lambdaλ设置得越大，多项式朴素贝叶斯的精确性会越低（虽然影响不是非常大）。
    fit_prior : 布尔值, 可不填 (默认为True)
    是否学习先验概率P(Y=c)。如果设置为false，则所有的样本类别输出都有相同的类别先验概率。即认为每个标签类出现的概率是1 n _ c l a s s e s \frac1{n\_classes} 
    n_classes
    1
    ​	
     。
    class_prior：形似数组的结构，结构为(n_classes, )，可不填（默认为None）
    类的先验概率P(Y=c)。如果没有给出具体的先验概率则自动根据数据来进行计算。
    布尔参数fit_prior表示是否要考虑先验概率，如果是False，则所有的样本类别输出都有相同的类别先验概率。否则，可以用第三个参数class_prior输入先验概率，或者不输入第三个参数class_prior让
    MultinomialNB自己从训练集样本来计算先验概率，此时的先验概率为P(Y=Ck)=mk/m。其中m为训练集样本总数量，mk为输出为第k个类别的训练集样本数。
     '''
#设置训练集
    mlt.fit(x_train, y_train)
    joblib.dump(tf,'saved_model/tf.pkl')
    joblib.dump(mlt,'saved_model/mlt.pkl')
    test_01=['奇才对骑士控卫唱主角 基德接班人降临阿联需猛醒 新浪体育讯在刚刚结束的华盛顿奇才主场迎战克里夫兰骑士的比赛中，奇才以102-107输掉了这场背靠背之战。状元秀约翰-沃尔上场43分钟，14投5中得到13分4篮板10助攻1抢断1盖帽，但也有6次失误，表现中规中矩。本场比赛，奇才对阵骑士，乍一看','中学生赴美交流要测“情感毒药”昨天，2011-2012年度“中学生赴美年度交流项目”(简称AYP项目)首次遴选活动在南京举行，来自全省六七十名“尖子生”进行了笔试和面试，获选者有机会赴美进行为期一年的交流。据了解，AYP项目是江苏省教育厅主管的卓越国际交流教育基金会所引进的非营利学生项目，旨在加强中美学生之间的国际交流。昨天上午10点10分，参加完美国初中生英语水平测试(SLEP)的考生进入了特别的考核环节。“马上考试官会在你们闭上眼睛以后偷偷指定你们其中的一个同学，在你们睁开眼睛']
    t2=[]
    for i in test_01:#注意python的for循环用法
        n= jieba.lcut(str(i))	#精确模式分词，适合做文本分析，装换成string形式
        t2.append(' '.join(n)) # 将每个标题的分词结果用空格连接起来
        j=j+1
        print(j)
    test_01=t2
    test_01=tf.transform(test_01)
    print(test_01)
    y_predict = mlt.predict(test_01)
    print(y_predict)
    print('y_predict')
    print('successful')
    return None

if __name__ =="__main__":
    fenlei()

