# 原始配置
import copy, math, os, pickle, time, pandas as pd, numpy as np, scipy.stats as ss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score

import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K

# from utils import *
# 加入官网的配置
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedKFold # 分集并5折交叉验证
from sklearn.preprocessing import StandardScaler

# 01-读入原始数据
ROOT_DIR = 'G:/Project/sepsis3_mortality_master/mort_icu_master'
DATA_FILEPATH = ROOT_DIR + '/data_mimiciv_icd9'
# RAW_DATA_FILEPATH = '../Extract_output_nogrouping/all_hourly_data.h5'
# GAP_TIME          = 6  # In hours
WINDOW_SIZE       = 24 # In hours
SEED              = 1
ID_COLS           = ['subject_id', 'hadm_id', 'icustay_id']
# GPU               = '2'
# GPU = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = GPU
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 准备数据集
# data_full_lvl2 = pd.read_hdf(DATA_FILEPATH, 'vitals_labs')
statics        = pd.read_csv(DATA_FILEPATH + '/static_data.csv')
statics = statics.set_index(['subject_id', 'hadm_id', 'stay_id'])
icd = pd.read_hdf(DATA_FILEPATH + '/C.h5')  # 这里的ICD诊断代码没有剔除第二次以后的，所以需要和statics合并，重新生成研究人群的ICD
statics['icd9_codes'] = icd['icd9_codes']
ICD = pd.DataFrame(statics['icd9_codes'])

# 从单级CCS分类文件中获取ICD到CCS的映射
def icdMapper(ccsFolder):
    #singleFile = open(os.path.join(ccsFolder, '$dxref 2015.csv'), 'r')   # 单级CCS文本
    singleFile = open(ccsFolder,'r')
    cnt = 0
    icd2Cid = {}        # len(icd2Cid)=15072
    cidSet = set()      # len(cidSet)=283
    singleFile.readline()
    singleFile.readline()
    singleFile.readline()
    for l in singleFile:    # ICD9,CCS,CCS描述,ICD9描述（,OptionalCCS,OptionalCCS描述）---procedure(diagnoses)
        tokens = l.strip().split(',')
        icd = tokens[0][1:-1].strip()   # ICD9
        curCid = int(tokens[1][1:-1].strip())   # ccs
        icd2Cid[icd] = curCid
    singleFile.close()
    return icd2Cid  # 返回的是ICD9:CCS单级编码映射的字典

# 读取ccs文件
ccsFolder = '../ccs'
icd2Cid = icdMapper(ccsFolder+'/$dxref 2015.csv')   # 单级ccs映射

def icdMappers(icds):
    ccs=[]
    for icd in icds:
        ccs.append(icd2Cid[icd.strip()])
    return ccs
to_ccs = lambda x: icdMappers(x)

ICD['ccs_codes'] = (ICD['icd9_codes']).apply(to_ccs)

def toIndex(lawSeqs):
    types = {}
    newSeqs = []
    for patients in lawSeqs:
        newPa = []
        for code in patients:
            if type(code) == str:
                code = code.strip()
            if code in types:
                newPa.append(types[code])
            else:
                types[code] = len(types)
                newPa.append(types[code])
        newSeqs.append(newPa)
    return types,newSeqs

'''
    ccs相关的：seqs(原始的),types（原始到索引的映射字典）,newSeqs（以单级ccs的索引为主的ccs诊断序列）
    icd9相关的：seqs_icd（原始的）,types_icd（原始到索引的映射字典）,newSeqs_icd（以ICD9诊断代码为主的序列）
'''
seqs_icd = ICD['icd9_codes'].values
seqs = ICD['ccs_codes'].values
types={}
types_icd={}
newSeqs = []
newSeqs_icd =[]
types,newSeqs = toIndex(seqs)
types_icd,newSeqs_icd = toIndex(seqs_icd)
ICD['ccs']=newSeqs
ICD['icd9'] = newSeqs_icd

'''
    二、build tree
    输入：多级CCS文件，原始的ICD9序列，ICDtype序列（即 ）
    输出：各个层级文件 .level*.pk
'''
inf = ccsFolder+'/ccs_multi_dx_tool_2015.csv'

outFile = '../mimic4/sepsis3'
# 读文件
infd = open(inf,'r')
_ = infd.readline()
# seqs_icd
# types_icd
startSet = set(types_icd.keys())
hitList = []
missList = []
cat1count = 0
cat2count = 0
cat3count = 0
cat4count = 0
for line in infd:  # ICD9,CCS1,CCS1Label,CCS2,CCS2Label,CCS3,CCS3Label,CCS4,CCS4Label
    tokens = line.strip().split(',')
    icd9 = tokens[0][1:-1].strip()
    cat1 = 'A_' + tokens[1][1:-1].strip()
    desc1 = 'A_' + tokens[2][1:-1].strip()
    cat2 = 'A_' + tokens[3][1:-1].strip()
    desc2 = 'A_' + tokens[4][1:-1].strip()
    cat3 = 'A_' + tokens[5][1:-1].strip()
    desc3 = 'A_' + tokens[6][1:-1].strip()
    cat4 = 'A_' + tokens[7][1:-1].strip()
    desc4 = 'A_' + tokens[8][1:-1].strip()

    # if icd9.startswith('E'):
    #     if len(icd9) > 4: icd9 = icd9[:4] + '.' + icd9[4:]
    # else:
    #     if len(icd9) > 3: icd9 = icd9[:3] + '.' + icd9[3:]
#     icd9 = 'D_' + icd9

    if icd9 not in types_icd:
        missList.append(icd9)   # 筛选的患者的诊断代码中没有的ICD9代码
    else:
        hitList.append(icd9)    # 筛选患者的诊断代码在分类代码中出现的

    if cat1 not in types_icd:
        cat1count += 1
        types_icd[cat1] = len(types_icd)

    if cat2 not in types_icd and cat2 != 'A_':
        cat2count += 1
        types_icd[cat2] = len(types_icd)

    if cat3 not in types_icd and cat3 != 'A_':
        cat3count += 1
        types_icd[cat3] = len(types_icd)

    if cat4 not in types_icd and cat4 != 'A_':
        cat4count += 1
        types_icd[cat4] = len(types_icd)
infd.close()

rootCode = len(types_icd)   # 最顶层的根节点，其实是一个虚拟的结点
types_icd['A_ROOT'] = rootCode
print(rootCode)

print('cat1count: %d' % cat1count)

print('cat2count: %d' % cat2count)

print('cat3count: %d' % cat3count)

print('cat4count: %d' % cat4count)

print('Number of total ancestors: %d' % (cat1count + cat2count + cat3count + cat4count + 1))

# print 'hit count: %d' % len(set(hitList))
print('miss count: %d' % len(startSet - set(hitList)))

missSet = startSet - set(hitList)

fiveMap = {}
fourMap = {}
threeMap = {}
twoMap = {}
oneMap = dict([(types_icd[icd], [types_icd[icd], rootCode]) for icd in missSet])

infd = open(inf, 'r')
infd.readline()

for line in infd:   # ICD9,CCS1,CCS1Label,CCS2,CCS2Label,CCS3,CCS3Label,CCS4,CCS4Label
    tokens = line.strip().split(',')
    icd9 = tokens[0][1:-1].strip()
    cat1 = tokens[1][1:-1].strip()
    desc1 = 'A_' + tokens[2][1:-1].strip()
    cat2 = tokens[3][1:-1].strip()
    desc2 = 'A_' + tokens[4][1:-1].strip()
    cat3 = tokens[5][1:-1].strip()
    desc3 = 'A_' + tokens[6][1:-1].strip()
    cat4 = tokens[7][1:-1].strip()
    desc4 = 'A_' + tokens[8][1:-1].strip()

    # if icd9.startswith('E'):
    #     if len(icd9) > 4: icd9 = icd9[:4] + '.' + icd9[4:]
    # else:
    #     if len(icd9) > 3: icd9 = icd9[:3] + '.' + icd9[3:]
#     icd9 = 'D_' + icd9

    if icd9 not in types_icd: continue
    icdCode = types_icd[icd9]

    codeVec = []

    if len(cat4) > 0:
        code4 = types_icd['A_' + cat4]
        code3 = types_icd['A_' + cat3]
        code2 = types_icd['A_' + cat2]
        code1 = types_icd['A_' + cat1]
        fiveMap[icdCode] = [icdCode, rootCode, code1, code2, code3, code4]
    elif len(cat3) > 0:
        code3 = types_icd['A_' + cat3]
        code2 = types_icd['A_' + cat2]
        code1 = types_icd['A_' + cat1]
        fourMap[icdCode] = [icdCode, rootCode, code1, code2, code3]
    elif len(cat2) > 0:
        code2 = types_icd['A_' + cat2]
        code1 = types_icd['A_' + cat1]
        threeMap[icdCode] = [icdCode, rootCode, code1, code2]
    else:
        code1 = types_icd['A_' + cat1]
        twoMap[icdCode] = [icdCode, rootCode, code1]
# 需要用到的层级序列
# pickle.dump(fiveMap, open(outFile + '.level5.pk', 'wb'), -1)
# pickle.dump(fourMap, open(outFile + '.level4.pk', 'wb'), -1)
# pickle.dump(threeMap, open(outFile + '.level3.pk', 'wb'), -1)
# pickle.dump(twoMap, open(outFile + '.level2.pk', 'wb'), -1)
# # pickle.dump(oneMap, open(outFile + '.level1.pk', 'wb'), -1)
# pickle.dump(types_icd, open(outFile + '.types_icd', 'wb'), -1)    # 原始的是len为5066的序列（诊断代码部分的ICD9：索引），这里更新为len为5794的序列（ICD9+CCS：索引）

'''
3.生成tree.seqs和newTree.seqs  =>  process_treeseqs.py
input：患者ICD9诊断代码索引号序列，对应icd列，即newSeqs_icd；第二个输入是（ICD9+CCS:索引号）对应了上面生成的types_icd
中间输入：mimic3.level*.pk，对应上面生成的几个层级序列
output：mimic3_tree.seqs和mimic3_newTree.seqs
'''
from functools import reduce
def process_newTrees(dataseqs, tree_old):
    # leaf2tree = pickle.load(open('../mimic3/sepsis3.level5.pk', 'rb'))
    # trees_l4 = pickle.load(open('../mimic3/sepsis3.level4.pk', 'rb'))
    # trees_l3 = pickle.load(open('../mimic3/sepsis3.level3.pk', 'rb'))
    # trees_l2 = pickle.load(open('../mimic3/sepsis3.level2.pk', 'rb'))
    leaf2tree = fiveMap
    trees_l4 = fourMap
    trees_l3 = threeMap
    trees_l2 = twoMap
    tree_seq = []

    leaf2tree.update(trees_l4)
    leaf2tree.update(trees_l3)
    leaf2tree.update(trees_l2)

    for patient in dataseqs:
        newPatient = []
        for code in patient:
            # newCode = []
            # for code in visit:
            if code in leaf2tree[code]:
                leaf2tree[code].remove(code)
                newPatient.append(leaf2tree[code])
            else:  # 表明code已经在上一个if中被删掉
                newPatient.append(leaf2tree[code])
        newPatient = list(set(reduce(lambda x,y:x+y, newPatient)))  # reduce将多个数组合并
        tree_seq.append(newPatient)

    newTreeseq = []
    for patient in tree_seq:
        newPatient = []
        for code in patient:
            # newVisit = []
            # for code in visit:
            #     newVisit.append(tree_old[code])
            newPatient.append(tree_old[code])
        newTreeseq.append(newPatient)

    return tree_seq,newTreeseq

outFile1 = '../mimic4/sepsis3_tree'
outFile2 = '../mimic4/sepsis3_newTree'

# data_seqs = pickle.load(open('../mimic3/sepsis3.seqs_icd', 'rb'))
# trees_type = pickle.load(open('../mimic3/sepsis3.types_icd', 'rb'))
data_seqs = newSeqs_icd
trees_type = types_icd
retype = dict([(v, k) for k, v in trees_type.items()])

treenode = {}  # 索引：CCS多级编码 映射祖先节点新数组和节点ccs_multi的信息
tree2old = {}  # 多级CCS就索引：新索引，映射祖先新数组与旧数组的index对应关系
count = 0

# 共有728个祖先节点(包含root节点和ccs分类节点)
for i in range(5066, len(retype)):
    treenode[count] = retype[i]
    tree2old[i] = count
    count += 1

treeSeq, newTreeSeq = process_newTrees(data_seqs, tree2old)  # 下标范围从0到727, 共728个节点
# treeSeq = process_Trees(data_seqs)  # 下标范围从4880到5607, 共728个节点

# pickle.dump(treeSeq, open(outFile1 + '.seqs', 'wb'), -1)
# pickle.dump(newTreeSeq, open(outFile2 + '.seqs', 'wb'), -1)

'''
4.生成邻接矩阵
输入：层级序列，（ICD+CCS：索引号）对应本文的types_icd
输出：.adj，edgelist.txt，.graph
'''
import dgl
import networkx as nx
import scipy.sparse as sp
def tree_levelall():
    # leaf2tree = pickle.load(open('../mimic3/sepsis3.level5.pk', 'rb'))
    # trees_l4 = pickle.load(open('../mimic3/sepsis3.level4.pk', 'rb'))
    # trees_l3 = pickle.load(open('../mimic3/sepsis3.level3.pk', 'rb'))
    # trees_l2 = pickle.load(open('../mimic3/sepsis3.level2.pk', 'rb'))
    leaf2tree = fiveMap
    trees_l4 = fourMap
    trees_l3 = threeMap
    trees_l2 = twoMap

    leaf2tree.update(trees_l4)
    leaf2tree.update(trees_l3)
    leaf2tree.update(trees_l2)

    return leaf2tree

# types = pickle.load(open('../mimic3/sepsis3.types_icd', 'rb'))
types_ = types_icd
retype = dict([(v, k) for k, v in types_.items()])

edgelist = {}
tree = tree_levelall()

for key, value in tree.items():
    for index, node in enumerate(value):
        if index == 0:
            continue
        if index == len(value) - 1:
            if node not in edgelist:
                edgelist[node] = [key]
            else:
                edgelist[node].append(key)
        else:
            if node not in edgelist:
                edgelist[node] = [value[index+1]]
            elif value[index+1] not in edgelist[node]:
                edgelist[node].append(value[index+1])

# Source nodes for edges
src_ids = []
# Destination nodes for edges
dst_ids = []

for key, value in edgelist.items():
    src_ids.extend([key for i in range(len(value))])
    dst_ids.extend(value)

temp = src_ids + dst_ids
# temp.extend(dst_ids)
code = list(set(temp))
all = [ i for i in range(5794)]
miss = list(set(all).difference(set(code)))
misscode = []
for i in miss:
    misscode.append(retype[i])

srcc = []
dstt = []
for code in misscode:
    tokens = code.strip().split('.')
    prefix = ''
    if len(tokens)==1:
        prefix = prefix + tokens[0]
    else:
        for i in range(len(tokens)-1):
            if i == 0 or i == len(tokens)-1:
                prefix = prefix + tokens[i]
            else:
                prefix = prefix + '.' + tokens[i]
    srcc.append(int(types_[prefix]))
    dstt.append(int(types_[code]))

src_ids.extend(srcc)
dst_ids.extend(dstt)

# pickle.dump(src_ids, open(outFile + '.src_ids', 'wb'), -1)
# pickle.dump(dst_ids, open(outFile + '.dst_ids', 'wb'), -1)
# # 保存要用的信息src_ids, dst_ids，直接使用torch版本处理保存adj和graph
# with open(outFile + '_protocol4.src_ids', 'wb') as file:
#     pickle.dump(src_ids, file, protocol=4)
# with open(outFile + '_protocol4.dst_ids', 'wb') as file:
#     pickle.dump(dst_ids, file, protocol=4)

g = dgl.graph((src_ids, dst_ids))
# 有向图转为无向图
bg = dgl.to_bidirected(g)
# print(g)

bbg = dgl.add_self_loop(bg)
# 1.保存sepsis3.graph
# pickle.dump(bbg, open(outFile + '.graph', 'wb'), -1)
# nx.draw_networkx(g)
# visual(bg)

nx_G = bg.to_networkx().to_undirected()
N = len(nx_G)
adj = nx.to_numpy_array(nx_G)
adj = sp.coo_matrix(adj)
# 2.保存sepsis3.adj
# pickle.dump(adj, open(outFile + '.adj', 'wb'), -1)


# 制作node2vec识别的edgelist
# 字典中的key值即为csv中列名
edgedata = pd.DataFrame({'src': src_ids, 'dst': dst_ids})

# 3.保存edgelist.txt
# with open('../mimic4/edgelist.txt', 'a+', encoding='utf-8') as f:
#     for line in edgedata.values:
#         f.write((str(line[0]) + '\t' + str(line[1]) + '\n'))

print('Done it!')

'''
5.生成图标签，参考process_graphLabel.py
Input:ccs_multi_dx_tool_2015.csv,sepsis3.seqs_icd,sepsis3.types_icd
Output: graphLabel.npy
'''
'''
原处理代码，改它
# outFile = '../mimic3/sepsis3'
# multidx = '../resource/ccs_multi_dx_tool_2015.csv'
# seqs = pickle.load(open('../resource/mimic3.seqs', 'rb'))  # newSeqs_icd
# types = pickle.load(open('../resource/mimic3.types', 'rb'))   # types_icd
# retype = dict(sorted([(v, k) for k, v in types.items()]))   # retype
# 将多级分类中的icd-9编码按照多级ccs分组
# ref = {}  # icd2Cid
# infd = open(multidx, 'r')
# infd.readline()
# for line in infd:
#     tokens = line.strip().replace('\'', '').split(',')
#     icd9 = 'D_' + convert_to_icd9(tokens[0].replace(' ', ''))
#     multiccs = int(tokens[1].replace(' ', ''))
#     ref[icd9] = multiccs
# infd.close()

# types中0-4879为icd9编码
# category = {}
# for i in range(4880):
#     category[i] = ref[retype[i]]
# 
# for i in range(4880,len(types)):
#     category[i] = convert_num(retype[i])
# 
# c1 = []
# for k in category.items():
#     c1.append(k)
# labels = np.array(c1)
# # idx_features_labels = np.loadtxt("{}{}.content".format("../model/data/cora/", "cora"), dtype=np.dtype(str))
# # labels = idx_features_labels[:, -1]
# classes = set(labels[:,-1])
# classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
# labels_onehot = np.array(list(map(classes_dict.get, labels[:,-1])), dtype=np.int32)
# # np.save('../resource/graphLabel', labels_onehot)
'''
# outFile = outFile
# multidx = inf
# newSeqs_icd
# types_icd
# retype
# icd2Cid 这个是单级ccs映射
# 我们需要的是多级ccs映射

def convert_num(dxStr):
    if len(dxStr) == 1: return int(dxStr[:])
    else :
        if dxStr[:2]=='A_': return int(0)
        return int(dxStr[:2].replace('.', ''))

# 将多级分类中的icd-9编码按照多级ccs分组
ref = {}
infd = open(inf, 'r')
infd.readline()
for line in infd:
    tokens = line.strip().replace('\'', '').split(',')
    icd9 = tokens[0].replace(' ','')
    # icd9 = 'D_' + convert_to_icd9(tokens[0].replace(' ', ''))
    multiccs = int(tokens[1].replace(' ', ''))
    ref[icd9] = multiccs
infd.close()

# types中0-5065为icd9编码
category = {}
for i in range(5066):
    category[i] = ref[retype[i]]

for i in range(5066,len(types_icd)):
    category[i] = convert_num(retype[i])

c1 = []
for k in category.items():
    c1.append(k)
labels = np.array(c1)
# idx_features_labels = np.loadtxt("{}{}.content".format("../model/data/cora/", "cora"), dtype=np.dtype(str))
# labels = idx_features_labels[:, -1]
classes = set(labels[:,-1])
classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
labels_onehot = np.array(list(map(classes_dict.get, labels[:, -1])), dtype=np.int32)
# np.save('../mimic4/graphLabel', labels_onehot)
print('Done it! Get graphLabel.npy')

'''
6.训练gcn_emb
得到嵌入向量-新建一个新文件夹embedding
'''