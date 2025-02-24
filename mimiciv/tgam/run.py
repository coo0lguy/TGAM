import time
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import Callback
import heapq
import operator
import os
from collections import defaultdict
import shap

# 原始配置
import copy, math, os, pickle, time, pandas as pd, numpy as np, scipy.stats as ss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, brier_score_loss

from keras import backend as K

from utils import *
# 加入官网的配置
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold  # 分集并5折交叉验证
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# gpu按内存所需分配
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 设置画布大小和绘制基础设置
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

WINDOW_SIZE = 24  # In hours
SEED = 1
# ID_COLS           = ['subject_id', 'hadm_id', 'icustay_id']
# GPU               = '2'
GPU = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
np.random.seed(SEED)
tf.random.set_seed(SEED)

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=50,
    mode='auto',
    restore_best_weights=True)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9
)

# 01-读入数据
ROOT_DIR = 'G:/Project/sepsis3_mortality_master/sepsis_mort_icu_master'
DATA_FILEPATH = ROOT_DIR + '/data/all_hourly_data.h5'
# RAW_DATA_FILEPATH = '../Extract_output_nogrouping/all_hourly_data.h5'
# GAP_TIME          = 6  # In hours
WINDOW_SIZE = 24  # In hours
SEED = 1
ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']
# GPU               = '2'
GPU = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
np.random.seed(SEED)
tf.random.set_seed(SEED)

if __name__ == '__main__':
    # 00读取数据
    #folder_data = '../data_fold5'    # 本机
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 构建相对路径
    folder_data = os.path.join(script_dir, "data_fold5")
    fold_trainSet = pickle.load(
        open(os.path.join(folder_data, 'fold_Set.train'), 'rb'))  # (X,structNoGCB,ICD,tree,struct)
    fold_testSet = pickle.load(open(os.path.join(folder_data, 'fold_Set.test'), 'rb'))
    fold_validSet = pickle.load(open(os.path.join(folder_data, 'fold_Set.valid'), 'rb'))
    fold_Xmean = pickle.load(open(os.path.join(folder_data, 'fold_Set.Xmean'), 'rb'))
    fold_Y = pickle.load(open(os.path.join(folder_data, 'fold_Set.labels'), 'rb'))  # (y_train,y_vali)
    y_test = pickle.load(open(os.path.join(folder_data, 'test.labels'), 'rb'))

    # gcn Embedding
    # gcn_emb = pickle.load(open('C:\\Users\\wangyi\\PycharmProjects\\sepsis_mort_icu_master\\mimiciv\\mimic4\\gcn_emb_onehot.emb', 'rb'))
    gcn_emb = pickle.load(open(
        'C:\\Users\\wangyi\\PycharmProjects\\sepsis_mort_icu_master\\mimiciv\\mimic4\\gcn_emb_onehot.emb',
        'rb'))
    diagcode_emb = gcn_emb[0][:5066]
    knowledge_emb = gcn_emb[0][5066:]

    TEARCAM_list = {
        'tra_out_dims': 74,
        'num_hiddens': 74,
        "norm_shape": [1, 2],
        "ffn_num_hiddens": 64,
        "num_heads": 8,
        "num_layers": 1,
        "dropout": 0.5
    }
    hyperparams_list = TEARCAM_list

    # 使用保存好的数据进行训练
    batch_size = 32
    epochs = 100
    class_weight = {0: 1, 1: 10}
    # 这个保存
    folder_name = "savedModels/tegcam"
    # training
    predictions = []
    predictions_valid_set = []
    predictions_training_set = []
    # ground_truth = []
    ground_truth_valid_set = []
    ground_truth_training_set = []
    fold_history = []

    i = 0
    for i in range(5):
        print(f'Fold {i}. Positive examples in validation: {np.sum(fold_Y[i][1])}...')

        # ICD9诊断代码序列，知识树序列
        icd, tree = padMatrix(fold_trainSet[i][2], fold_trainSet[i][3])  # 这里会有问题，因为维度和原始的是不一样的，需要改改这个函数
        icd_valid, tree_valid = padMatrix(fold_validSet[i][2], fold_validSet[i][3])
        icd_test, tree_test = padMatrix(fold_testSet[i][2], fold_testSet[i][3])

        # 静态不含共存病的数据
        train_sta = np.repeat(np.expand_dims(fold_trainSet[i][1], 1), 24, 1).reshape([-1, 1, 24, 31])
        valid_sta = np.repeat(np.expand_dims(fold_validSet[i][1], 1), 24, 1).reshape([-1, 1, 24, 31])
        test_sta = np.repeat(np.expand_dims(fold_testSet[i][1], 1), 24, 1).reshape([-1, 1, 24, 31])

        ground_truth_training_set.append(fold_Y[i][0])  # 5折中训练集真实标签
        ground_truth_valid_set.append(fold_Y[i][1])  # 5折中验证集真实标签

        X_mean = fold_Xmean[i]
        base_params = {'input_size': X_mean.shape[2], 'diagcode_emb': diagcode_emb, 'knowledge_emb': knowledge_emb}
        # #     base_params = {'X_mean':np.where(np.isnan(X_mean),0,X_mean),'input_size':X_mean.shape[2]}
        model_hyperparams = copy.copy(base_params)
        model_hyperparams.update(
            {k: v for k, v in hyperparams_list.items() if k in (
                'tra_out_dims', 'num_hiddens', 'norm_shape', 'ffn_num_hiddens', 'num_heads', 'num_layers', 'dropout')}
        )
        model = TEGCAM(**model_hyperparams, batch_size=batch_size,X_mean=np.where(np.isnan(X_mean),0,X_mean) , in_icdSize=(icd.shape[1],), in_treeSize=(tree.shape[1],),
                       in_structSize=(1, train_sta.shape[2], train_sta.shape[3]))
        
        model.compile(
            optimizer='adam',
            loss=[focal_loss(alpha=0.25, gamma=1.0)],
            metrics=METRICS)
        
        exponent_lr = ExponentDecayScheduler(learning_rate_base=0.001,
                                             global_epoch_init=0,
                                             decay_rate=0.9,
                                             min_learn_rate=1e-6
                                             )

    
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            filepath=f"C:\\Users\\wangyi\\PycharmProjects\\sepsis_mort_icu_master\\mimiciv\\tegcam\\savedModels\\tegcam/kfold{i}_best4",
            monitor='val_loss',
            save_best_only=True)

       
        files = [f for f in os.listdir(
            "C:\\Users\\wangyi\\PycharmProjects\\sepsis_mort_icu_master\\mimiciv\\tegcam\\savedModels\\tegcam")
                 if
                 f.endswith(f"kfold{i}_best4")]

        print(files)
        
        model = keras.models.load_model(
            f"C:\\Users\\wangyi\\PycharmProjects\\sepsis_mort_icu_master\\mimiciv\\tegcam\\savedModels\\tegcam/{files[0]}",
            custom_objects={'focal_loss_fixed': focal_loss})
        # model = keras.models.load_model(f"./{folder_name}/{files[0]}")
        predictions_training_set.append(model.predict((fold_trainSet[i][0], [icd, tree], train_sta)))
        predictions_valid_set.append(model.predict((fold_validSet[i][0], [icd_valid, tree_valid], valid_sta)))
        predictions.append(model.predict((fold_testSet[i][0], [icd_test, tree_test], test_sta)))  # 测试结果


    
    from sklearn.metrics import f1_score, brier_score_loss

    Sen = []
    Spe = []
    Auc = []
    Auprc = []
    Brier_score = []
    # sen,spe,acc,auc,auprc = 0,0,0,0,0
    for i in range(5):
        fpr, tpr, thresholds = roc_curve(y_test, predictions[i])
        youden_index = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[youden_index]
        sen = tpr[youden_index]
        spe = 1 - fpr[youden_index]

        fold_cm = confusion_matrix(y_test, predictions[i].round())
        tn, fp, fn, tp = fold_cm[0][0], fold_cm[0][1], fold_cm[1][0], fold_cm[1][1]
        # sen = tp / (tp + fn)
        # spe = tn / (tn + fp)
        auc = roc_auc_score(y_test, predictions[i])
        auprc = average_precision_score(y_test, predictions[i])
        brier_score = brier_score_loss(y_test, predictions[i])
        Sen.append(sen)
        Spe.append(spe)
        Auc.append(auc)
        Auprc.append(auprc)
        Brier_score.append(brier_score)
    print('TEGCAM:')
    print(r'Sen:%0.4f $\pm$ %0.4f' % (np.mean(Sen), np.std(Sen)))
    print('Spe:%0.4f $\pm$ %0.4f' % (np.mean(Spe), np.std(Spe)))
    print('Auc:%0.6f $\pm$ %0.4f' % (np.mean(Auc), np.std(Auc)))
    print('Auprc:%0.4f $\pm$ %0.4f' % (np.mean(Auprc), np.std(Auprc)))
    print('Brier score:%0.4f $\pm$ %0.4f' % (np.mean(Brier_score), np.std(Brier_score)))

    
