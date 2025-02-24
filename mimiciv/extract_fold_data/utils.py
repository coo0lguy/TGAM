import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K

import copy, math, os, pickle, time, pandas as pd, numpy as np, scipy.stats as ss
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score

def to_3D_tensor(df):
    idx = pd.IndexSlice
    return np.dstack((df.loc[idx[:,:,:,i], :].values for i in sorted(set(df.index.get_level_values('hours_in')))))


def prepare_dataset(df,statics, Ys,batch_size, shuffle=True):
    """
    dfs = (df_train, df_dev, df_test).
    df_* = (subject, hadm, icustay, hours_in) X (level2, agg fn \ni {mask, mean, time})
    Ys_series = (subject, hadm, icustay) => label.
    """
#     batch_size = Ys.shape[0]
    Xd = tf.convert_to_tensor(to_3D_tensor(df).astype(np.float32))
    # Xs = tf.convert_to_tensor(statics.values.astype(np.float32))
    # label = tf.convert_to_tensor(Ys.values.astype(np.int64))
    dataset = tf.data.Dataset.from_tensor_slices((Xd, statics, Ys))
    if shuffle:
        return dataset.shuffle(len(Xd[0])).batch(batch_size)
    else:
        return dataset.batch(batch_size)
    # return utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

# 绘制损失图
def plot_loss(history, label, n):
    # Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
                 color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
                 color=colors[n], label='Val ' + label,
                 linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()

# 查看训练历史记录
def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend();

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))

# 绘制ROC
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(fp, tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

# 绘制AUPRC
def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
