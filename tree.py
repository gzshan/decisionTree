#coding=utf-8
"""
决策树算法：ID3
按照属性的不同取值构建判定树，将不同类别的样本分割开
算法关键在于每一次如何选取一个最优的划分属性使得不同类的样本尽可能分开
主要方法：信息增益、信息增益率、基尼指数，分别对应算法为：ID3,C4.5，CART
"""
import os
import math
import operator
import pickle
from treePlotter import createPlot

"""计算一个样本集合的香农熵，公式：-Plog2P"""
def calcShannonEnt(dataSet):
	numEntries = len(dataSet) #样本个数
	labelCounts = {} #dict，用以存储每一类样本的个数，（类别：数量）
	
	"""依次遍历每一个样本，计算每一类的数量"""
	for featVec in dataSet:
		currentLabel = featVec[-1]  #样本最后一列为其label值
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel]=0
		labelCounts[currentLabel] += 1 #统计每一类数量

	print labelCounts
	"""根据公式计算熵"""
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries #这一类的概率
		shannonEnt -= prob*math.log(prob,2) #公式计算
	
	"""返回熵值"""
	return shannonEnt

"""创建一个数据集合，嵌套列表的形式，其中，每一个样例的最后一个值为其对应的标签"""
def createDataSet():
	dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
	labels = ['no surfacing','flippers'] #属性的集合
	return dataSet,labels

"""划分数据集，将所有属性axis对应的值为value的样本返回"""
def splitDataSet(dataSet,axis,value):
	retDataSet = []
	"""遍历每一个样本，判断其属性axis对应的值是否为value"""
	for featVec in dataSet:
		if featVec[axis] == value: #若满足条件，则将这些样本中对应的这一属性去掉，也就是已经用过这一属性了
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec) #所有满足条件的样本构成一个集合返回
	return retDataSet

"""选择一个最优的划分属性，使信息增益最大"""
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0])-1 #属性数量
	baseEntropy = calcShannonEnt(dataSet) #样本整体的信息熵
	bestInfoGain = 0.0 #最大的信息增益
	bestFeature = -1 #最优的划分属性
	
	"""依次遍历所有的属性，计算其信息增益"""
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet] #当前属性的所有可能的取值
		uniqueVals = set(featList) #set 去掉重复

		newEntrop = 0.0 #划分之后的信息熵
		for value in uniqueVals: #对于该属性的每一个可能的取值（也就是划分之后的每一个分支）
			subDataSet = splitDataSet(dataSet,i,value) #获取该分支所有的样本
			prob = len(subDataSet) / float(len(dataSet)) #概率
			newEntrop += prob * calcShannonEnt(subDataSet) #根据公式计算新的经验条件熵

		infoGain = baseEntropy - newEntrop #获得的信息增益
		if infoGain>bestInfoGain: #选取最大的信息增益对应的属性
			bestInfoGain = infoGain
			bestFeature = i
	"""返回最优属性"""
	return bestFeature

"""投票选出其中得票数最高的一个类别，当属性全都用完后，标记为得票数最多的类别"""
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys:
			classCount[vote] = 0
		classCount[vote] += 1
	
	#按票数排序
	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)

	return sortedClassCount[0][0]

"""创建决策树，递归实现，ID3算法"""
def createTree(dataSet,labels):
	classList = [example[-1] for example in dataSet] #当前所有的类别
	"""递归结束条件1：所有的样本都属于同一类，无需划分"""
	if classList.count(classList[0]) == len(classList): 
		return classList[0]
	
	""""递归结束条件2：所有的属性都用完"""
	if len(dataSet[0])==1:
		return majorityCnt(classList)
	
	"""选取当前的最优划分属性，进行划分"""
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	
	myTree = {bestFeatLabel:{}}  #将该结点标记为最优属性，准备创建分支
	del(labels[bestFeat]) #删除该属性

	featValues = [example[bestFeat] for example in dataSet] #所有的属性值
	uniqueVals = set(featValues)

	for value in uniqueVals:    #对每一个属性值创建一个分支
		subLabels = labels[:]
		myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels) #递归进行划分
	
	"""返回学习到的决策树"""
	return myTree

"""使用决策树进行测试"""
def classify(inputTree,featLabels,testVec):
	firstStr = inputTree.keys()[0] #根结点
	secondDict = inputTree[firstStr] #子树
	featIndex = featLabels.index(firstStr) #找该属性对应的序号

	for key in secondDict.keys(): #遍历子树，判断属于哪一分支
		if testVec[featIndex] == key: 
			if type(secondDict[key]).__name__=='dict': #该结点属于分支结点
				classLabel = classify(secondDict[key],featLabels,testVec)
			else: #叶子结点
				classLabel = secondDict[key]
	return classLabel

"""存储决策树"""
def storeTree(inputTree,fileName):
	fw = open(fileName,'w')
	pickle.dump(inputTree,fw)
	fw.close()

"""从磁盘加载决策树"""
def grabTree(fileName):
	fr = open(fileName)
	return pickle.load(fr)


if __name__ == '__main__':
	myDat,labels = createDataSet()
	#myDat[0][-1] = 'maybe'
	#entropy = calcShannonEnt(myDat)
	mytree = createTree(myDat,labels)
	print mytree
	createPlot(mytree)
	storeTree(mytree,"./tree.model")