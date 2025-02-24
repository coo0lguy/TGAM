import tensorflow as tf
from tensorflow import keras as keras
from keras import backend as K
from keras import regularizers
from keras.layers import Input,Dense,Dropout,Flatten,BatchNormalization,Conv2D,Masking,MultiHeadAttention,concatenate,GRU,LSTM
from keras.models import Sequential,Model
from keras.optimizers import Adam

import copy, math, os, pickle, time, pandas as pd, numpy as np, scipy.stats as ss
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix
from tfkan.layers import DenseKAN, Conv2DKAN

def exponent(global_epoch,
            learning_rate_base,
            decay_rate,
            min_learn_rate=0,
            ):

    learning_rate = learning_rate_base * pow(decay_rate, global_epoch)
    learning_rate = max(learning_rate,min_learn_rate)
    return learning_rate

class ExponentDecayScheduler(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度，指数型下降
    """
    def __init__(self,
                 learning_rate_base,
                 decay_rate,
                 global_epoch_init=0,
                 min_learn_rate=0,
                 verbose=0):
        super(ExponentDecayScheduler, self).__init__()
        # 基础的学习率
        self.learning_rate_base = learning_rate_base
        # 全局初始化epoch
        self.global_epoch = global_epoch_init

        self.decay_rate = decay_rate
        # 参数显示
        self.verbose = verbose
        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.min_learn_rate = min_learn_rate
        self.learning_rates = []

    def on_epoch_end(self, epochs ,logs=None):
        self.global_epoch = self.global_epoch + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)
	#更新学习率
    def on_epoch_begin(self, batch, logs=None):
        lr = exponent(global_epoch=self.global_epoch,
                    learning_rate_base=self.learning_rate_base,
                    decay_rate = self.decay_rate,
                    min_learn_rate = self.min_learn_rate)
        K.set_value(self.model.optimizer.lr, lr)
        # 更新学习率
        # self.model.optimizer.learning_rate.assign(lr)


        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_epoch + 1, lr))



def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0,
                             min_learn_rate=0,
                             ):
    """
    参数：
            global_step: 上面定义的Tcur，记录当前执行的步数。
            learning_rate_base：预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降。
            total_steps: 是总的训练的步数，等于epoch*sample_count/batch_size,(sample_count是样本总数，epoch是总的循环次数)
            warmup_learning_rate: 这是warm up阶段线性增长的初始值
            warmup_steps: warm_up总的需要持续的步数
            hold_base_rate_steps: 这是可选的参数，即当warm up阶段结束后保持学习率不变，知道hold_base_rate_steps结束后才开始学习率下降
    """
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                            'warmup_steps.')
    #这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
        (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    #如果hold_base_rate_steps大于0，表明在warm up结束后学习率在一定步数内保持不变
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                    learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                                'warmup_learning_rate.')
        #线性增长的实现
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        #只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                    learning_rate)

    learning_rate = max(learning_rate,min_learn_rate)
    return learning_rate


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度, 余弦退火更新版衰减
    """
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 min_learn_rate=0,
                 # interval_epoch代表余弦退火之间的最低点
                 interval_epoch=[0.05, 0.15, 0.30, 0.50],
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        # 基础的学习率
        self.learning_rate_base = learning_rate_base
        # 热调整参数
        self.warmup_learning_rate = warmup_learning_rate
        # 参数显示
        self.verbose = verbose
        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.min_learn_rate = min_learn_rate
        self.learning_rates = []

        self.interval_epoch = interval_epoch
        # 贯穿全局的步长
        self.global_step_for_interval = global_step_init
        # 用于上升的总步长
        self.warmup_steps_for_interval = warmup_steps
        # 保持最高峰的总步长
        self.hold_steps_for_interval = hold_base_rate_steps
        # 整个训练的总步长
        self.total_steps_for_interval = total_steps

        self.interval_index = 0
        # 计算出来两个最低点的间隔
        self.interval_reset = [self.interval_epoch[0]]
        for i in range(len(self.interval_epoch)-1):
            self.interval_reset.append(self.interval_epoch[i+1]-self.interval_epoch[i])
        self.interval_reset.append(1-self.interval_epoch[-1])

	#更新global_step，并记录当前学习率
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        self.global_step_for_interval = self.global_step_for_interval + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

	#更新学习率
    def on_batch_begin(self, batch, logs=None):
        # 每到一次最低点就重新更新参数
        if self.global_step_for_interval in [0]+[int(i*self.total_steps_for_interval) for i in self.interval_epoch]:
            self.total_steps = self.total_steps_for_interval * self.interval_reset[self.interval_index]
            self.warmup_steps = self.warmup_steps_for_interval * self.interval_reset[self.interval_index]
            self.hold_base_rate_steps = self.hold_steps_for_interval * self.interval_reset[self.interval_index]
            self.global_step = 0
            self.interval_index += 1

        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps,
                                      min_learn_rate = self.min_learn_rate)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))

codeCount = 5066  # icd9数
treeCount = 728  # 分类树的祖先节点数量

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_recall_curve, roc_curve, auc

def cross_validation_plot_roc_curve(ground_truth, predictions, filename=None):
    plt.figure(figsize=(15, 10))
    i = 0
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    for g, p in zip(ground_truth, predictions):
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_true=g, y_score=p, drop_intermediate=False)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.4f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    #     mean_auc = auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    if filename:
        plt.savefig(filename, dpi=500)
    plt.show()


# 绘制AUPRC


def cross_validation_plot_prc_curve(ground_truth, predictions, filename=None):
    plt.figure(figsize=(15, 10))
    i = 0
    mean_precision = np.linspace(0, 1, 100)
    recalls = []  # 干啥的
    auprcs = []
    for g, p in zip(ground_truth, predictions):
        # Compute PRC curve and area the curve
        precision, recall, thresholds = precision_recall_curve(g, p)
        recalls.append(np.interp(mean_precision, precision, recall))  #
        recalls[-1][0] = 1.0  #
        prc_auc = average_precision_score(g, p)
        auprcs.append(prc_auc)
        plt.plot(precision, recall, lw=1, alpha=0.3,
                 label='PRC fold %d (AUPRC = %0.4f)' % (i, prc_auc))
        i += 1

    plt.plot([0, 1], [1, 0], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_recall = np.mean(recalls, axis=0)
    mean_recall[-1] = 0.0
    mean_auprc = np.mean(auprcs)
    std_auprc = np.std(auprcs)
    plt.plot(mean_precision, mean_recall, color='b',
             label=r'Mean PRC (AUPRC = %0.4f $\pm$ %0.4f)' % (mean_auprc, std_auprc),
             lw=2, alpha=.8)

    std_recall = np.std(recalls, axis=0)
    recalls_upper = np.minimum(mean_recall + std_recall, 1)
    recalls_lower = np.maximum(mean_recall - std_recall, 0)
    plt.fill_between(mean_precision, recalls_lower, recalls_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    if filename:
        plt.savefig(filename, dpi=500)
    plt.show()

import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix', filename=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(7, 7))
    plt.gca().grid(False)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if filename:
        plt.savefig(filename, dpi=500)
    plt.show()



def padMatrix(seqs, treeseqs=''):
    # lengths = np.array([len(seq) for seq in seqs]) - 1
    n_samples = len(seqs)
    # maxlen = np.max(lengths)

    x = np.zeros((n_samples, codeCount), dtype=np.int8)

    if len(treeseqs) > 0:
        tree = np.zeros((n_samples, treeCount), dtype=np.int8)
        for idx, (seq, tseq) in enumerate(zip(seqs, treeseqs)):
            x[idx, :][seq[:]] = 1
            tree[idx, :][tseq[:]] = 1
            # for xvec, subseq in zip(x[idx, :], seq[:-1]):
            #     xvec[subseq] = 1.
            # for tvec, subseq in zip(tree[idx, :, :], tseq[:-1]):
            #     tvec[subseq] = 1.
        return x, tree

    else:
        for idx, seq in enumerate(seqs):
            x[idx, :][seq[:]] = 1
            # for xvec, subseq in zip(x[idx, :, :], seq[:-1]):
            #     xvec[subseq] = 1.
        return x


class ScaledDotProductAttention(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.output_dim = output_dim
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(2, input_shape[0][2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[1][2], self.output_dim),
                                 initializer='uniform',
                                 trainable=True)

        super(ScaledDotProductAttention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        Lt, rnn_ht = inputs
        WQ = K.dot(rnn_ht, self.W)
        WK = K.dot(Lt, self.kernel[0])
        WV = K.dot(Lt, self.kernel[1])
        # WQ.shape (None, 41, 128)

        # 转置 K.permute_dimensions(WK, [0, 2, 1]).shape (None, 128, 41)
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64 ** 0.5)

        weights = K.softmax(QK)
        # QK.shape (None, 41, 41)
        context_vector = K.batch_dot(weights, WV)

        return context_vector, weights

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config

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

class cross_attention(keras.layers.Layer):
    def __init__(self,heads,key_dim,**kwargs):
        super(cross_attention, self).__init__(**kwargs)
        self.cross_att = keras.layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim)

    def call(self, att1, att2, *args, **kwargs):
        x = att1
        y = att2
        x = tf.expand_dims(x,axis=1)
        y = tf.expand_dims(y,axis=1)
        a1 = self.cross_att(x,y)
        a2 = self.cross_att(y,x)
        return tf.concat([a1[:,0,:],a2[:,0,:]],-1)


class self_attention(keras.layers.Layer):
    def __init__(self,heads,key_dim,**kwargs):
        super(self_attention, self).__init__(**kwargs)
        # self.num_heads = heads
        # self.key_dim = key_dim
        self.self_att = keras.layers.MultiHeadAttention(num_heads=heads,key_dim=key_dim)

    def call(self, inputs, *args, **kwargs):
        x = tf.expand_dims(inputs,axis=1)
        att = self.self_att(x,x)
        return att[:,0,:]

# 04-建模01
# 定义一个求延迟的gamma层--结果比较好的一个网络，先保留*****
class gammaLayer(keras.layers.Layer):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(gammaLayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def build(self, input_shape):
        self.kernel = self.add_weight("gamma_kernel",
                                      shape=[int(input_shape[-1]), self.out_features],
                                      initializer=tf.random_normal_initializer())
        if self.bias == True:
            self.bias = self.add_weight("gamma_bias",
                                        shape=[self.out_features], initializer=tf.zeros_initializer())

    def call(self, inputs):
        '''
        inputs:时间间隔，shape(batch_size,timesteps,features)-(batch_size,24,31)
        '''
        return K.exp(-K.relu(tf.matmul(inputs, self.kernel) + self.bias))


# 定义一个三个权重的层，用于计算重置门、更新门、还有h_telda
class grudLayer(keras.layers.Layer):
    def __init__(self, in_features, hidden_size, bias=True, **kwargs):
        super(grudLayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.bias = bias

    def build(self, input_shape):
        self.kernel_x = self.add_weight("kernel_x_hat",
                                        shape=[int(input_shape[-1]), self.hidden_size],
                                        initializer=tf.random_normal_initializer())
        self.kernel_h = self.add_weight("kernel_h_t-1",
                                        shape=[self.hidden_size, self.hidden_size],
                                        initializer=tf.random_normal_initializer())
        self.kernel_m = self.add_weight("kernel_m",
                                        shape=[int(input_shape[-1]), self.hidden_size],
                                        initializer=tf.random_normal_initializer())
        if self.bias == True:
            self.bias = self.add_weight("bias",
                                        shape=[self.hidden_size], initializer=tf.zeros_initializer())

    def call(self, x, h_, m):
        return tf.matmul(x, self.kernel_x) + tf.matmul(h_, self.kernel_h) + tf.matmul(m, self.kernel_m) + self.bias


#@save
class PositionalEncoding(keras.layers.Layer):
    """位置编码"""
    def __init__(self,input_size, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = keras.layers.Dropout(dropout)
        # 创建一个足够长的P
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        # self.P[:, :, 0::2] = np.sin(X)
        # self.P[:, :, 1::2] = np.cos(X)

        self.gamma_x = gammaLayer(input_size, input_size)
        self.num_hiddens = num_hiddens

    def call(self, X, **kwargs):
        # print(type(X))
        X, X_last_obsv, Mask, Delta = X
        delta_x = self.gamma_x(Delta)
        X = Mask * X + (1.0 - Mask) * (delta_x * X_last_obsv)
        X = X * tf.math.sqrt(tf.cast(self.num_hiddens,dtype=tf.float32))
        # print(type(X))
        # X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)

#@save
class PositionWiseFFN(keras.layers.Layer):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super().__init__(*kwargs)
        self.dense1 = keras.layers.Dense(ffn_num_hiddens)
        self.relu = keras.layers.ReLU()
        self.dense2 = keras.layers.Dense(ffn_num_outputs)

    def call(self, X):
        return self.dense2(self.relu(self.dense1(X)))

#@save
class AddNorm(keras.layers.Layer):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = keras.layers.Dropout(dropout)
        self.ln = keras.layers.LayerNormalization(normalized_shape)

    def call(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y, **kwargs) + X)

# transformer 编码器部分的编码块
#@save
class EncoderBlock(keras.layers.Layer):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.attention = keras.layers.MultiHeadAttention(num_heads, key_size, value_size, dropout, bias)
        # self.attention = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
        #                                         num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def call(self, X, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, **kwargs), **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)

class Xembed(keras.layers.Layer):
    def __init__(self,input_size,**kwargs):
        super(Xembed, self).__init__(**kwargs)
        # self.delta_size =
        self.gamma_x = gammaLayer(input_size,input_size)

    def call(self, inputs, *args, **kwargs):
        X, X_last_obsv, Mask, Delta = inputs
        delta_x = self.gamma_x(Delta)
        X = Mask * X + (1.0 - Mask) * (delta_x * X_last_obsv)
        return X

#@save
class TransformerEncoder(keras.Model):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hiddens, num_heads,
                 num_layers, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        # self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        # self.embedding = Xembed(vocab_size)
        self.pos_encoding = PositionalEncoding(vocab_size, num_hiddens, dropout)
        self.blks = [EncoderBlock(
            key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_hiddens, num_heads, dropout, bias) for _ in range(
            num_layers)]

    def call(self, X, **kwargs):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        # X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(
        #     tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        X = self.pos_encoding(X, **kwargs)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, **kwargs)
            self.attention_weights[
                i] = blk.attention
                # i] = blk.attention.attention.attention_weights
        return X

# # model2:把最初的想法实现
# class TEGCAM(keras.Model):
#     def __init__(self,diagcode_emb,knowledge_emb,in_icdSize,in_treeSize,in_structSize,input_size,tra_out_dims,num_hiddens,norm_shape,ffn_num_hiddens,num_heads,num_layers,dropout,bias=False):
#         super(TEGCAM,self).__init__()
#         self.in_grud = input_size
#         self.in_icd = in_icdSize
#         self.in_tree = in_treeSize
#         self.in_struct = in_structSize
#         self.trans_encoder = TransformerEncoder(input_size,input_size,input_size,input_size,num_hiddens,norm_shape,ffn_num_hiddens,num_heads,num_layers,dropout,bias)
#         self.sdp_Attention = ScaledDotProductAttention(output_dim=tra_out_dims)
#         # self.grud = GRUD(input_size=self.in_grud,cell_size=cell_size,hidden_size=hidden_size,X_mean=X_mean,output_last=output_last,batch_size=batch_size)
#         self.dence_64 = keras.layers.Dense(256,activation='relu')
#         # self.dence_50 = keras.layers.Dense(32,activation='sigmoid')
# #         self.dence_1 = keras.layers.Dense(1,activation='sigmoid')
#         self.dence_11 = keras.layers.Dense(256,activation='relu')
#         self.drop = keras.layers.Dropout(0.5)
#         self.self_att_ = self_attention(4, 64)
#         self.self_att = self_attention(4,64)
#         self.cross_att = cross_attention(4, 64)
#         # self.cross_att = keras.layers.MultiHeadAttention(4,32)
#
#         self.fc = keras.layers.Dense(1, activation='sigmoid',name='fc3',kernel_regularizer=regularizers.l2(0.01))
#         self.sigmoid = tf.keras.layers.Activation('sigmoid')  # Sigmoid激活层
#         self.kanfc = DenseKAN(1, grid_size=3)
#         self.bn = keras.layers.BatchNormalization()
#
#         # icd
#         self.mask = keras.layers.Masking(mask_value=0)
#         self.emb = keras.layers.Dense(128,activation='relu', kernel_initializer=keras.initializers.constant(diagcode_emb), name='emb')
#         self.icd_rnn = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True, dropout=0.5))
#         self.icd_att = ScaledDotProductAttention(output_dim=256)
#         self.tree_emb = keras.layers.Dense(128, kernel_initializer=keras.initializers.constant(knowledge_emb),trainable=True,name='tree_emb')
#         self.tree_att = ScaledDotProductAttention(output_dim=128)
#
#         # self.stru_in = keras.layers.Input(shape=self.in_struct,name='input_struct')
#         self.conv1kan = Conv2DKAN(filters=128, kernel_size=(3,1), strides=(1,1), padding='same', kan_kwargs={'grid_size': 3})
#         self.conv2kan = Conv2DKAN(filters=64, kernel_size=(3,1), strides=(1,1), padding='same', kan_kwargs={'grid_size': 3})
#         self.conv1 = keras.layers.Conv2D(128,(3,1),strides=(1,1),padding='same',name='24h_conv',activation='relu')
#         self.pool = keras.layers.MaxPooling2D(pool_size=(1,3),padding='same',name='24h_pooled')
#         self.drop1 = keras.layers.Dropout(0.5,name='dropout1')
#         self.conv2 = keras.layers.Conv2D(64, (3, 1), strides=(1, 1), padding='same', name='conv', activation='relu')
#         self.pool2 = keras.layers.MaxPooling2D(pool_size=(1, 3), padding='same', name='pooled')
#         self.drop2 = keras.layers.Dropout(0.3, name='dropout2')
#         self.bn1 = keras.layers.BatchNormalization(axis=3,name='bn1')
#         self.flat = keras.layers.Flatten()
#         self.fc1 = keras.layers.Dense(256,activation='relu',name='fc1',kernel_regularizer=regularizers.l2(0.01))
#         self.drop3 = keras.layers.Dropout(0.2,name='dropout3')
#         self.bn2 = keras.layers.BatchNormalization(axis=1,name='bn2')
#
#         # self.rnn = keras.layers.GRU(num_hiddens,return_sequences=True)
#         self.rnn = keras.layers.LSTM(num_hiddens, return_sequences=True)
#
# #         self.drop = keras.layers.Dropout(0.5)
#
#     def call(self, x):
#         in_encoder, in_icd, in_struct = x
#         _,_,_,delta=in_encoder   # x,last_x,mask,delta
#         # out_grud = self.grud(in_grud)
#         out_trans_encoder = self.trans_encoder(in_encoder)
#         # print('---------------------')
#         out_gru = self.rnn(delta)
#         out_tear, weights = self.sdp_Attention([out_trans_encoder,out_gru])
#         # out_gru = self.rnn(out_trans_encoder)
#         # print(out_gru.shape)
#
#         # icd
#         mask = self.mask(in_icd[0])
#         icd_emb = self.emb(mask)
#         icd_emb = tf.tile(tf.expand_dims(icd_emb, axis=1), multiples=[1,3,1])
#         rnn = self.icd_rnn(icd_emb)
#         head1, weight1 = self.icd_att([rnn,rnn])
#         head2, weight2 = self.icd_att([rnn,rnn])
#         tree_mask = self.mask(in_icd[1])
#         tree_emb = self.tree_emb(tree_mask)
#         tree_emb = tf.tile(tf.expand_dims(tree_emb,axis=1),multiples=[1,3,1])
#         head1, weight1 = self.tree_att([tree_emb,head1])
#         head2, weight2 = self.tree_att([tree_emb,head2])
#         st = tf.concat([head1,head2],-1)
#
#
#         # statics
#         # out_struct = self.conv1(in_struct)
#         out_struct = self.conv1kan(in_struct)
#         out_struct = self.pool(out_struct)
#         out_struct = self.drop1(out_struct)
#         # out_struct = self.conv2(out_struct)
#         out_struct = self.conv2kan(out_struct)
#         out_struct = self.pool2(out_struct)
#         out_struct = self.drop2(out_struct)
#         out_struct = self.bn1(out_struct)
#         out_struct = self.flat(out_struct)
#         out_struct = self.fc1(out_struct)
#         out_struct = self.drop3(out_struct)
#         out_struct = self.bn2(out_struct)
#         # in_struct = keras.layers.Input(shape=self.in_struct)
#         # out_struct = self.dence_64(in_struct)
#         # out_struct = self.drop(out_struct)
#         out_struct = self.dence_64(out_struct)
#         # out_struct = self.drop(out_struct)
#         # out_struct = self.dence_1(out_struct)
#
#         vv_att = self.self_att_(out_tear)
#         vv_att = self.flat(vv_att)
#         vv_att = self.dence_11(vv_att)
#         tt_att = self.self_att(out_struct)
#         vv_att = tf.tile(tf.expand_dims(vv_att,axis=1), multiples=[1,3,1])
#         tt_att = tf.tile(tf.expand_dims(tt_att,axis=1), multiples=[1,3,1])
#
#         vt_att = self.cross_att(vv_att,tt_att)
#         vi_att = self.cross_att(vv_att,st)
#         ti_att = self.cross_att(tt_att,st)
#
#         merged = tf.concat([vt_att,vi_att,ti_att],-1)
#         merged =self.flat(merged)
#         # a1 = tf.squeeze(merged,1)
#         # print(a1.shape,a2.shape,out_struct.shape,out_grud.shape)
#         # merged = tf.concat([a1,out_struct],1)
#         # output = self.dence_11(merged)
#         # return self.fc(self.drop(self.bn(merged)))
#         return self.sigmoid(self.kanfc(self.drop(self.bn(merged))))
#         # return out_grud

# model3:按照画好的模型图进行搭建
class TERAM(keras.Model):
    def __init__(self,diagcode_emb,knowledge_emb,in_icdSize,in_treeSize,in_structSize,input_size,tra_out_dims,num_hiddens,norm_shape,ffn_num_hiddens,num_heads,num_layers,dropout,bias=False):
        super(TERAM,self).__init__()
        self.in_grud = input_size
        self.in_icd = in_icdSize
        self.in_tree = in_treeSize
        self.in_struct = in_structSize
        self.trans_encoder = TransformerEncoder(input_size,input_size,input_size,input_size,num_hiddens,norm_shape,ffn_num_hiddens,num_heads,num_layers,dropout,bias)
        self.sdp_Attention = ScaledDotProductAttention(output_dim=tra_out_dims)
        # self.grud = GRUD(input_size=self.in_grud,cell_size=cell_size,hidden_size=hidden_size,X_mean=X_mean,output_last=output_last,batch_size=batch_size)
        self.dence_64 = keras.layers.Dense(256,activation='relu')
        # self.dence_50 = keras.layers.Dense(32,activation='sigmoid')
#         self.dence_1 = keras.layers.Dense(1,activation='sigmoid')
        self.dence_11 = keras.layers.Dense(256,activation='relu')
        self.drop = keras.layers.Dropout(0.5)
        # self.self_att_ = self_attention(4, 64)
        self.self_att = self_attention(4, 64)
        self.cross_att = cross_attention(4, 64)
        # self.cross_att = keras.layers.MultiHeadAttention(4,32)

        self.fc = keras.layers.Dense(1, activation='sigmoid',name='fc3',kernel_regularizer=regularizers.l2(0.01))
        self.bn = keras.layers.BatchNormalization()

        # icd
        self.mask = keras.layers.Masking(mask_value=0)
        self.emb = keras.layers.Dense(128,activation='relu', kernel_initializer=keras.initializers.constant(diagcode_emb), name='emb')
        # self.icd_rnn = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True, dropout=0.5))
        self.icd_att = self_attention(2, 256)
        self.tree_emb = keras.layers.Dense(128, kernel_initializer=keras.initializers.constant(knowledge_emb),trainable=True,name='tree_emb')
        self.tree_att = ScaledDotProductAttention(output_dim=128)
        # self.tree_att = keras.layers.MultiHeadAttention(num_heads=2, key_dim=128)
        self.dence_icd = keras.layers.Dense(256,activation='relu', kernel_regularizer=regularizers.l2(0.01))

        # self.stru_in = keras.layers.Input(shape=self.in_struct,name='input_struct')
        self.conv1 = keras.layers.Conv2D(128,(3,1),strides=(1,1),padding='same',name='24h_conv',activation='relu')
        self.pool = keras.layers.MaxPooling2D(pool_size=(1,3),padding='same',name='24h_pooled')
        self.drop1 = keras.layers.Dropout(0.5,name='dropout1')
        # self.conv2 = keras.layers.Conv2D(64, (3, 1), strides=(1, 1), padding='same', name='conv', activation='relu')
        # self.pool2 = keras.layers.MaxPooling2D(pool_size=(1, 3), padding='same', name='pooled')
        # self.drop2 = keras.layers.Dropout(0.3, name='dropout2')
        self.bn1 = keras.layers.BatchNormalization(axis=3,name='bn1')
        self.flat = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(256,activation='relu',name='fc1',kernel_regularizer=regularizers.l2(0.01))
        # self.drop3 = keras.layers.Dropout(0.2,name='dropout3')
        # self.bn2 = keras.layers.BatchNormalization(axis=1,name='bn2')

        self.rnn = keras.layers.GRU(num_hiddens,return_sequences=True)

#         self.drop = keras.layers.Dropout(0.5)

    def call(self, x):
        in_encoder, in_icd, in_struct = x
        _,_,_,delta=in_encoder   # x,last_x,mask,delta
        # out_grud = self.grud(in_grud)
        out_trans_encoder = self.trans_encoder(in_encoder)
        # print('---------------------')
        out_gru = self.rnn(delta)
        out_tear, weights = self.sdp_Attention([out_trans_encoder,out_gru])
        # out_gru = self.rnn(out_trans_encoder)
        # print(out_gru.shape)

        # icd
        mask = self.mask(in_icd[0])
        icd_emb = self.emb(mask)
        # icd_emb = tf.tile(tf.expand_dims(icd_emb, axis=1), multiples=[1,3,1])
        # rnn = self.icd_rnn(icd_emb)
        icd_att = self.icd_att(icd_emb)
        # head2, weight2 = self.icd_att([icd_emb,icd_emb])
        tree_mask = self.mask(in_icd[1])
        tree_emb = self.tree_emb(tree_mask)
        # tree_emb = tf.tile(tf.expand_dims(tree_emb,axis=1),multiples=[1,3,1])
        icd_att = tf.expand_dims(icd_att,axis=1)
        tree_emb = tf.expand_dims(tree_emb,axis=1)
        tree_att, weight1 = self.tree_att([tree_emb,icd_att])
        # head2, weight2 = self.tree_att([tree_emb,head2])
        st = self.flat(tree_att)
        st = self.dence_icd(st)


        # statics
        out_struct = self.conv1(in_struct)
        out_struct = self.pool(out_struct)
        out_struct = self.drop1(out_struct)
        # out_struct = self.conv2(out_struct)
        # out_struct = self.pool2(out_struct)
        # out_struct = self.drop2(out_struct)
        out_struct = self.bn1(out_struct)
        out_struct = self.flat(out_struct)
        out_struct = self.fc1(out_struct)
        # out_struct = self.drop3(out_struct)
        # out_struct = self.bn2(out_struct)

        out_struct = self.dence_64(out_struct)

        # vv_att = self.self_att_(out_tear)
        vv_att = self.flat(out_tear)
        vv_att = self.dence_11(vv_att)
        tt_att = self.self_att(out_struct)
        # vv_att = tf.tile(tf.expand_dims(vv_att,axis=1), multiples=[1,3,1])
        # tt_att = tf.tile(tf.expand_dims(tt_att,axis=1), multiples=[1,3,1])

        # 交叉注意力，输入输出都是（None,**)
        vt_att = self.cross_att(vv_att,tt_att)
        vi_att = self.cross_att(vv_att,st)
        ti_att = self.cross_att(tt_att,st)

        merged = concatenate([vt_att,vi_att,ti_att])
        # merged =self.flat(merged)
        # a1 = tf.squeeze(merged,1)
        # print(a1.shape,a2.shape,out_struct.shape,out_grud.shape)
        # merged = tf.concat([a1,out_struct],1)
        # output = self.dence_11(merged)
        return self.fc(self.drop(self.bn(merged)))
        # return out_grud

# model4：把model3的交叉注意力直接拼接训练
class TERAM4(keras.Model):
    def __init__(self,diagcode_emb,knowledge_emb,in_icdSize,in_treeSize,in_structSize,input_size,tra_out_dims,num_hiddens,norm_shape,ffn_num_hiddens,num_heads,num_layers,dropout,bias=False):
        super(TERAM4,self).__init__()
        self.in_grud = input_size
        self.in_icd = in_icdSize
        self.in_tree = in_treeSize
        self.in_struct = in_structSize
        self.trans_encoder = TransformerEncoder(input_size,input_size,input_size,input_size,num_hiddens,norm_shape,ffn_num_hiddens,num_heads,num_layers,dropout,bias)
        self.sdp_Attention = ScaledDotProductAttention(output_dim=tra_out_dims)
        self.dence_64 = keras.layers.Dense(64,activation='relu')  # 128
        self.dence_11 = keras.layers.Dense(64,activation='relu')   # 128
        self.drop = keras.layers.Dropout(0.5)
        self.self_att = self_attention(4, 64)

        self.fc = keras.layers.Dense(1, activation='sigmoid',name='fc3',kernel_regularizer=regularizers.l2(0.01))
        self.bn = keras.layers.BatchNormalization()

        # icd
        self.mask = keras.layers.Masking(mask_value=0)
        self.emb = keras.layers.Dense(128,activation='relu', kernel_initializer=keras.initializers.constant(diagcode_emb), name='emb')
        self.icd_att = self_attention(2, 128)
        self.tree_emb = keras.layers.Dense(128, kernel_initializer=keras.initializers.constant(knowledge_emb),trainable=True,name='tree_emb')
        self.tree_att = ScaledDotProductAttention(output_dim=128)
        self.dence_icd = keras.layers.Dense(64,activation='relu', kernel_regularizer=regularizers.l2(0.01))

        self.conv1 = keras.layers.Conv2D(64,(3,1),strides=(1,1),padding='same',name='24h_conv',activation='relu')  # 128
        self.pool = keras.layers.MaxPooling2D(pool_size=(1,3),padding='same',name='24h_pooled')
        self.drop1 = keras.layers.Dropout(0.5,name='dropout1')
        self.bn1 = keras.layers.BatchNormalization(axis=3,name='bn1')
        self.flat = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(64,activation='relu',name='fc1',kernel_regularizer=regularizers.l2(0.01))

        self.rnn = keras.layers.GRU(num_hiddens,return_sequences=True)


    def call(self, x):
        in_encoder, in_icd, in_struct = x
        _,_,_,delta=in_encoder   # x,last_x,mask,delta
        out_trans_encoder = self.trans_encoder(in_encoder)
        # print('---------------------')
        out_gru = self.rnn(delta)
        out_tear, weights = self.sdp_Attention([out_trans_encoder,out_gru])

        # icd
        mask = self.mask(in_icd[0])
        icd_emb = self.emb(mask)
        icd_att = self.icd_att(icd_emb)
        tree_mask = self.mask(in_icd[1])
        tree_emb = self.tree_emb(tree_mask)
        icd_att = tf.expand_dims(icd_att,axis=1)
        tree_emb = tf.expand_dims(tree_emb,axis=1)
        tree_att, weight1 = self.tree_att([tree_emb,icd_att])
        st = self.flat(tree_att)
        st = self.dence_icd(st)


        # statics
        out_struct = self.conv1(in_struct)
        out_struct = self.pool(out_struct)
        out_struct = self.drop1(out_struct)
        out_struct = self.bn1(out_struct)
        out_struct = self.flat(out_struct)
        out_struct = self.fc1(out_struct)

        out_struct = self.dence_64(out_struct)

        # vv_att = self.self_att_(out_tear)
        vv_att = self.flat(out_tear)
        vv_att = self.dence_11(vv_att)
        tt_att = self.self_att(out_struct)

        # 交叉注意力，输入输出都是（None,**)

        merged = concatenate([vv_att,tt_att,st])
        return self.fc(self.drop(self.bn(merged)))




# 构建GRUD模型
class GRUD(keras.Model):
    def __init__(self, input_size, cell_size, hidden_size, X_mean, batch_size=0, output_last=False):
        super(GRUD, self).__init__()
        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size

        self.identity = tf.eye(input_size)
        self.X_mean = tf.Variable(tf.constant(X_mean))
        self.X_mean = tf.cast(self.X_mean, tf.float32)

        self.zl = grudLayer(self.delta_size, self.hidden_size)
        self.rl = grudLayer(self.delta_size, self.hidden_size)
        self.hl = grudLayer(self.mask_size, self.hidden_size)

        #         self.input = keras.layers.Dense(hidden_size)
        #         self.hidden = keras.layers.Dense(hidden_size)

        self.gamma_x_l = gammaLayer(self.delta_size, self.delta_size)  # 输出延迟x
        self.gamma_h_l = gammaLayer(self.delta_size, self.hidden_size)  # 输出延迟h

        self.output_last = output_last

        #         self.fc = keras.layers.Dense(1)
        self.fc = keras.layers.Dense(1, activation='sigmoid')
        self.bn = keras.layers.BatchNormalization()
        self.drop = keras.layers.Dropout(0.3)

    def step(self, x, x_last_obsv, x_mean, h, mask, delta):
        batch_size = x.shape[0]
        dim_size = x.shape[1]

        delta_x = self.gamma_x_l(delta)

        #         delta_x = tf.exp(-tf.reduce_max(self.zeros,gamma_x_l_delta))

        delta_h = self.gamma_h_l(delta)

        #         delta_h = tf.exp(-tf.reduce_max(self.zeros_h,gamma_h_l_delta))

        #         x_mean = tf.tile(x_mean,[batch_size,1])
        #         tf.cast(x_mean,tf.float32)
        #         print(mask.shape,delta_x.shape,x_last_obsv.shape,x_mean.shape)
        x = mask * x + (1.0 - mask) * (delta_x * x_last_obsv + (1.0 - delta_x) * x_mean)
        #         print(delta_h,delta_h.shape)
        #         print(h,h.shape)
        h = delta_h * h
        #         combined = tf.concat((x,h,mask),1)

        # 在这里加入dropout
        x_z = self.drop(x)
        x_r = self.drop(x)
        x_h = self.drop(x)
        m_z = self.drop(mask)
        m_r = self.drop(mask)
        m_h = self.drop(mask)
        h_z = self.drop(h)
        h_r = self.drop(h)


        z = tf.sigmoid(self.zl(x_z, h_z, m_z))
        r = tf.sigmoid(self.rl(x_r, h_r, m_r))
        #         combined_new = tf.concat((x,r*h,mask),1)
        h_tilde = tf.tanh(self.hl(x_h, r * h, m_h))
        h = (1 - z) * h + z * h_tilde

        return h

    def call(self, in_put):
        X, X_last_obsv, Mask, Delta = in_put
        #         batch_size = X.shape[0]
        #         print('go on!')
        #         print(X.shape[0],batch_size)
        step_size = X.shape[1]
        # spatial_size = X.shape[2]
        #         print('no')
        #         print(batch_size)
        Hidden_state = self.initHidden()
        #         print('yes')
        #         print(Hidden_state)
        outputs = None
        for i in range(step_size):
            Hidden_state = self.step(
                tf.squeeze(X[:, i:i + 1, :], 1),
                tf.squeeze(X_last_obsv[:, i:i + 1, :], 1),
                tf.squeeze(self.X_mean[:, i:i + 1, :], 1),
                Hidden_state,
                tf.squeeze(Mask[:, i:i + 1, :], 1),
                tf.squeeze(Delta[:, i:i + 1, :], 1)
            )
            if outputs is None:
                outputs = tf.expand_dims(Hidden_state, 1)
            else:
                outputs = tf.concat((tf.expand_dims(Hidden_state, 1), outputs), 1)

        #         return tf.sigmoid(self.drop(self.bn(self.fc(Hidden_state))))
        # return self.fc(self.drop(self.bn(Hidden_state)))  # 单独的时候用的是这个
        return Hidden_state

    def initHidden(self):
        Hidden_state = tf.zeros(self.hidden_size)
        #         recurrent_initializer='orthogonal'
        #         Hidden_state = keras.initializers.get(recurrent_initializer)
        #         tf.Variable(tf.zeros((batch_size,self.hidden_size)))
        return Hidden_state


class TEGCAM(keras.Model):
    def __init__(self,diagcode_emb,knowledge_emb,batch_size,X_mean,in_icdSize,in_treeSize,in_structSize,input_size,tra_out_dims,num_hiddens,norm_shape,ffn_num_hiddens,num_heads,num_layers,dropout,bias=False):
        super(TEGCAM,self).__init__()
        cell_size = 32
        hidden_size = 64
        output_last = True
        self.in_grud = input_size
        self.in_icd = in_icdSize
        self.in_tree = in_treeSize
        self.in_struct = in_structSize
        self.trans_encoder = TransformerEncoder(input_size,input_size,input_size,input_size,num_hiddens,norm_shape,ffn_num_hiddens,num_heads,num_layers,dropout,bias)
        self.sdp_Attention = ScaledDotProductAttention(output_dim=tra_out_dims)
        self.grud = GRUD(input_size=self.in_grud,cell_size=cell_size,hidden_size=hidden_size,X_mean=X_mean,output_last=output_last,batch_size=batch_size)
        self.dence_64 = keras.layers.Dense(256,activation='relu')
        #self.dence_50 = keras.layers.Dense(32,activation='sigmoid')         
        #self.dence_1 = keras.layers.Dense(1,activation='sigmoid')
        self.dence_11 = keras.layers.Dense(256,activation='relu')
        self.drop = keras.layers.Dropout(0.5)
        self.self_att_ = self_attention(4, 64)
        self.self_att = self_attention(4,64)
        self.cross_att = cross_attention(4, 64)
        # self.cross_att = keras.layers.MultiHeadAttention(4,32)

        self.fc = keras.layers.Dense(1, activation='sigmoid',name='fc3',kernel_regularizer=regularizers.l2(0.01))
        # self.sigmoid = tf.keras.layers.Activation('sigmoid')  # Sigmoid激活层
        self.bn = keras.layers.BatchNormalization()

        # icd
        self.mask = keras.layers.Masking(mask_value=0)
        self.emb = keras.layers.Dense(128,activation='relu', kernel_initializer=keras.initializers.constant(diagcode_emb), name='emb')
        self.icd_rnn = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True, dropout=0.5))
        self.icd_att = ScaledDotProductAttention(output_dim=256)
        self.tree_emb = keras.layers.Dense(128, kernel_initializer=keras.initializers.constant(knowledge_emb),trainable=True,name='tree_emb')
        self.tree_att = ScaledDotProductAttention(output_dim=128)

        # self.stru_in = keras.layers.Input(shape=self.in_struct,name='input_struct')

        self.conv1 = keras.layers.Conv2D(128,(3,1),strides=(1,1),padding='same',name='24h_conv',activation='relu')
        self.pool = keras.layers.MaxPooling2D(pool_size=(1,3),padding='same',name='24h_pooled')
        self.drop1 = keras.layers.Dropout(0.5,name='dropout1')
        self.conv2 = keras.layers.Conv2D(64, (3, 1), strides=(1, 1), padding='same', name='conv', activation='relu')
        self.pool2 = keras.layers.MaxPooling2D(pool_size=(1, 3), padding='same', name='pooled')
        self.drop2 = keras.layers.Dropout(0.3, name='dropout2')
        self.bn1 = keras.layers.BatchNormalization(axis=3,name='bn1')
        self.flat = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(256,activation='relu',name='fc1',kernel_regularizer=regularizers.l2(0.01))
        self.drop3 = keras.layers.Dropout(0.2,name='dropout3')
        self.bn2 = keras.layers.BatchNormalization(axis=1,name='bn2')

        # self.rnn = keras.layers.GRU(num_hiddens,return_sequences=True)
        # self.rnn = keras.layers.LSTM(num_hiddens, return_sequences=True)

#         self.drop = keras.layers.Dropout(0.5)

    def call(self, x):
        in_encoder, in_icd, in_struct = x
        _,_,_,delta=in_encoder   # x,last_x,mask,delta
        #out_gru = self.grud(in_encoder)
        out_tear = self.grud(in_encoder)
        #out_trans_encoder = self.trans_encoder(in_encoder)
        # # print('---------------------')
        # out_gru = self.rnn(delta)
        #out_tear, weights = self.sdp_Attention([out_trans_encoder,out_gru])
        # out_gru = self.rnn(out_trans_encoder)
        # print(out_gru.shape)

        # icd
        mask = self.mask(in_icd[0])
        icd_emb = self.emb(mask)
        icd_emb = tf.tile(tf.expand_dims(icd_emb, axis=1), multiples=[1,3,1])
        rnn = self.icd_rnn(icd_emb)
        head1, weight1 = self.icd_att([rnn,rnn])
        head2, weight2 = self.icd_att([rnn,rnn])
        tree_mask = self.mask(in_icd[1])
        tree_emb = self.tree_emb(tree_mask)
        tree_emb = tf.tile(tf.expand_dims(tree_emb,axis=1),multiples=[1,3,1])
        head1, weight1 = self.tree_att([tree_emb,head1])
        head2, weight2 = self.tree_att([tree_emb,head2])
        st = tf.concat([head1,head2],-1)


        # statics
        out_struct = self.conv1(in_struct)

        out_struct = self.pool(out_struct)
        out_struct = self.drop1(out_struct)
        out_struct = self.conv2(out_struct)

        out_struct = self.pool2(out_struct)
        out_struct = self.drop2(out_struct)
        out_struct = self.bn1(out_struct)
        out_struct = self.flat(out_struct)
        out_struct = self.fc1(out_struct)
        out_struct = self.drop3(out_struct)
        out_struct = self.bn2(out_struct)
        # in_struct = keras.layers.Input(shape=self.in_struct)
        # out_struct = self.dence_64(in_struct)
        # out_struct = self.drop(out_struct)
        out_struct = self.dence_64(out_struct)
        # out_struct = self.drop(out_struct)
        # out_struct = self.dence_1(out_struct)

        vv_att = self.self_att_(out_tear)
        vv_att = self.flat(vv_att)
        vv_att = self.dence_11(vv_att)
        tt_att = self.self_att(out_struct)
        vv_att = tf.tile(tf.expand_dims(vv_att,axis=1), multiples=[1,3,1])
        tt_att = tf.tile(tf.expand_dims(tt_att,axis=1), multiples=[1,3,1])

        vt_att = self.cross_att(vv_att,tt_att)
        vi_att = self.cross_att(vv_att,st)
        ti_att = self.cross_att(tt_att,st)

        merged = tf.concat([vt_att,vi_att,ti_att],-1)
        merged =self.flat(merged)
        # a1 = tf.squeeze(merged,1)
        # print(a1.shape,a2.shape,out_struct.shape,out_grud.shape)
        # merged = tf.concat([a1,out_struct],1)
        # output = self.dence_11(merged)
        # return self.fc(self.drop(self.bn(merged)))
        return self.fc(self.drop(self.bn(merged)))
