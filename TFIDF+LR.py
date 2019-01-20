import numpy as np
import pandas as pd
"""
在某些平台评论中会经常出现一些有毒评论（即一些粗鲁，不尊重或者可能让某人离开讨论的评论），
这使得许多人不愿意再表达自己并放弃在平台中评论。因此，为了促进用户对话，提出一系列的方案，来缓解这一问题。
我们将其看作一个文本分类问题，来介绍一系列的文本分类方案。
https://mp.weixin.qq.com/s/pXqaJ_gwdqRHto2zWqQwXg

1.1 评价指标:
每类标签的AUC的平均值，作为评价指标
1.2.
在这篇文章中，我将介绍最简单也是最常用的一种文本分类方法——从TFIDF中提取文本的特征，以逻辑回归作为分类器。
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('input/train.csv').fillna(' ')
test = pd.read_csv('input/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

#1、首先，我们利用TFIDF提取文本词语的信息：
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)  # 得到tf-idf矩阵，稀疏矩阵表示法
test_word_features = word_vectorizer.transform(test_text)   # 得到tf-idf矩阵，稀疏矩阵表示法

#2、为了充分表征文本信息，我们也提取文本字的ngram信息，我们将ngram设置为（2，6），
# 也就是说我们会最少提取两个字母作为单词的信息，最多会提取6个字母作为单词：
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
#3、接下来我们就开始合并文本词表征以及字表征：
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)
#4、所有的文本特征都提取完成后，我们就可以用机器学习的分类器——逻辑回归，
# 训练模型。这是一个多标签问题，我们将其看作6个二分类问题求解，即我们假设两两标签是没有关系的。
train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')

    #通过传入的模型，训练三次，最后将三次结果求平均值
    #交叉验证优点：
    # 1：交叉验证用于评估模型的预测性能，尤其是训练好的模型在新数据上的表现，可以在一定程度上减小过拟合。
    # 2：还可以从有限的数据中获取尽可能多的有效信息。
    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))  #sklearn的cross_val_score进行交叉验证
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))
    #交叉验证训练后，做预测
    classifier.fit(train_features, train_target)#训练模型
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))
submission.to_csv(submission.csv, index=False)
from sklearn.metrics import roc_auc_score
test_label=pd.read_csv('input/test_labels.csv')
# submission = pd.read_csv('input/submission.csv')
auc_sum = 0
for class_ in class_names:
    # print(test_label[test_label.id.isin(test_label[test_label[class_]==-1].id.tolist())==False])
    # print(test_label[class_] == -1)
    sub_test_label=test_label[test_label.id.isin(test_label[test_label[class_]==-1].id.tolist())==False]#取出所有有值的不为-1的 值 #
    sub_submission=submission[submission.id.isin(test_label[test_label[class_]==-1].id.tolist())==False]
    auc_sum += roc_auc_score(sub_test_label[class_],sub_submission[class_])
print("test_average_auc_score:",auc_sum/len(class_names))
#备注：如果test_label=-1,该样本的不计入auc的计算中。

#https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams

# 为什么还要使用ROC和AUC呢？
# 因为ROC曲线有个很好的特性：
# 当测试集中的正负样本的分布变化的时候，ROC曲线能够保持不变
# 。在实际的数据集中经常会出现类不平衡（class imbalance）现象，即负样本比正样本多很多（或者相反），而且测试数据中的正负样本的分布也可能随着时间变