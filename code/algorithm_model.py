# *- coding: utf-8 -*-
# @Author: bruce li
# @Date:   2018-11-18 13:39:45
# @Last Modified by:   bruce
# @Last Modified time: 2019-05-09 15:59:03
# @Email:	talkwithh@163.com

from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV

def SVM_model():
	# clf_svm = OneVsRestClassifier(SVC(C=0.99,kernel='linear'))
	clf_svm = LinearSVC(C=0.99)
	clf = CalibratedClassifierCV(clf_svm)
	# model_svm = clf_svm.fit(x_train,y_train)
	return clf

def bayes_model():
	clf = MultinomialNB(alpha=0.0001)
	# clf = ComplementNB(alpha=0.01)
	return clf

def knn_model():
	clf_knn = KNeighborsClassifier()
	# model_knn = clf_knn.fit(x_train,y_train)

	return clf_knn

def sgd_model():
	clf_sgd = SGDClassifier(alpha=.001,max_iter=50,loss='hinge')	# default loss=hinge
	clf = CalibratedClassifierCV(clf_sgd)
	# model_sgd = clf_sgd.fit(x_train,y_train)

	return clf

def forest_model():
	clf_forest = RandomForestClassifier(n_estimators=60,oob_score=True, random_state=10)
	# model_forest = clf_forest.fit(x_train,y_train)

	return clf_forest
