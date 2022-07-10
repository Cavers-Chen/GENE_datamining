import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input, Flatten, Conv2D, MaxPool2D
from keras.layers import Conv1D, MaxPooling1D, Embedding, merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split

labels_organ = []
labels = []
dataset = []
f_sets = []
f_num = 0

# df1 = pd.read_csv("GBM_fpkm_01.csv")
df1 = pd.read_csv("MYTEST.csv")
df1 = df1.drop(columns='gene_name')
df1 = df1.drop(columns='gene_type')
df1 = df1.drop(index=0)

LIST1 = df1.columns.values
LIST1 = np.array(LIST1)
LIST1 = LIST1[1:]
for i in range(len(LIST1)):
    # data = int(LIST1[i][5:7])
    if( LIST1[i][5:7] not in f_sets):
        f_num = f_num + 1
        f_sets.append(LIST1[i][5:7])
    labels_organ.append(f_num)

features = df1[df1.columns[0]]
print(features)
for j in range(len(df1.iloc[0])):
    if j != 0:
        data = df1[df1.columns[j]]
        data = [ float(x) for x in data ]
        data = np.array(data)
        data = data.reshape(60483, 1)
        dataset.append(data)
        #labels.append('GBM')
        labels.append('0')
        #labels_organ.append()

# df2 = pd.read_csv("LIHC_fpkm_01.csv")
# df2 = df2.drop(columns='gene_name')
# df2 = df2.drop(columns='gene_type')
# df2 = df2.drop(index=0)
# df2 = df2.loc[:,:]
#
# LIST2 = df2.columns.values
# LIST2 = np.array(LIST2)
# LIST2 = LIST2[1:]
# for i in range(len(LIST2)):
#     # data = LIST2[i][5:7]
#     # labels_organ.append(data)
#     if( LIST2[i][5:7] not in f_sets):
#         f_num = f_num + 1
#         f_sets.append(LIST2[i][5:7])
#     labels_organ.append(f_num)
#
# for j in range(len(df2.iloc[0])):
#     if j!=0 :
#         data = df2[df2.columns[j]]
#         data = [ float(x) for x in data ]
#         data = np.array(data)
#         data = data.reshape(60483, 1)
#         dataset.append(data)
#         #labels.append('LIHC')
#         labels.append('1')
#
# df3 = pd.read_csv("LUAD_fpkm_01.csv")
# df3 = df3.drop(columns='gene_name')
# df3 = df3.drop(columns='gene_type')
# df3 = df3.drop(index=0)
# df3 = df3.loc[:,:]
#
# LIST3 = df3.columns.values
# LIST3 = np.array(LIST3)
# LIST3 = LIST3[1:]
# for i in range(len(LIST3)):
#     # data = LIST3[i][5:7]
#     # labels_organ.append(data)
#     if( LIST3[i][5:7] not in f_sets):
#         f_num = f_num + 1
#         f_sets.append(LIST3[i][5:7])
#     labels_organ.append(f_num)
#
# for j in range(len(df3.iloc[0])):
#     if j!=0:
#         data = df3[df3.columns[j]]
#         data = [ float(x) for x in data ]
#         data = np.array(data)
#         data = data.reshape(60483, 1)
#         dataset.append(data)
#         #labels.append('LUAD')
#         labels.append('2')
#
# df4 = pd.read_csv("SKCM_fpkm_01.csv")
# df4 = df4.drop(columns='gene_name')
# df4 = df4.drop(columns='gene_type')
# df4 = df4.drop(index=0)
# df4 = df4.loc[:,:]
#
# LIST4 = df4.columns.values
# LIST4 = np.array(LIST4)
# LIST4 = LIST4[1:]
# for i in range(len(LIST4)):
#     # data = LIST4[i][5:7]
#     # labels_organ.append(data)
#     if( LIST4[i][5:7] not in f_sets):
#         f_num = f_num + 1
#         f_sets.append(LIST4[i][5:7])
#     labels_organ.append(f_num)
#
# for j in range(len(df4.iloc[0])):
#     if j!=0:
#         data = df4[df4.columns[j]]
#         data = [ float(x) for x in data ]
#         data = np.array(data)
#         data = data.reshape(60483,1)
#         dataset.append(data)
#         #labels.append('SKCM')
#         labels.append('3')

print(len(dataset))
print(len(labels_organ))
print(len(labels))
if len(dataset)==len(labels_organ):
    print(labels_organ)
    print(labels)

dataset = np.array(dataset)
labels = np.array(labels)
features = np.array(features)
labels = to_categorical(np.asarray(labels))
labels_organ_ = to_categorical(np.asarray(labels_organ))

X_train, X_test, y_train, y_test = train_test_split(dataset,labels_organ_,test_size=0.3)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(dataset.shape)
print(X_train.shape)

sequence_input = Input(shape=(60483, 1, ), dtype='float32')
# 5 level
l_cov1 = Conv1D(128, 7, activation='relu')(sequence_input)
l_pool1 = MaxPooling1D(16)(l_cov1)
l_cov2 = Conv1D(128, 7, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(32)(l_cov2)
l_cov3 = Conv1D(128, 7, activation='relu')(l_pool2)
l_pool3 = MaxPooling1D(108)(l_cov3)
l_flat = Flatten()(l_pool3)
# 全连接中间层
l_dense = Dense(128, activation='relu')(l_flat)
# 全连接输出层，softmax连接
preds = Dense( f_num+1, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
# 提交模型，设置自适应学习参数
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.summary()

model.fit(X_train, y_train,validation_data=(X_test, y_test), batch_size=32, epochs=10)

# 33/33 [==============================] - 83s 2s/step - loss: 68.7736 - acc: 0.6635 - val_loss: 2.6087 - val_acc: 0.8362
# Epoch 2/10
# 33/33 [==============================] - 74s 2s/step - loss: 5.6955 - acc: 0.7686 - val_loss: 4.3515 - val_acc: 0.7155
# Epoch 3/10
# 33/33 [==============================] - 78s 2s/step - loss: 2.9301 - acc: 0.8158 - val_loss: 2.6067 - val_acc: 0.8362
# Epoch 4/10
# 33/33 [==============================] - 73s 2s/step - loss: 1.1756 - acc: 0.8891 - val_loss: 6.0966 - val_acc: 0.5862
# Epoch 5/10
# 33/33 [==============================] - 74s 2s/step - loss: 1.2256 - acc: 0.9084 - val_loss: 0.5561 - val_acc: 0.9138
# Epoch 6/10
# 33/33 [==============================] - 72s 2s/step - loss: 0.8732 - acc: 0.9122 - val_loss: 0.4386 - val_acc: 0.9914
# Epoch 7/10
# 33/33 [==============================] - 75s 2s/step - loss: 1.0898 - acc: 0.9325 - val_loss: 4.2481 - val_acc: 0.8017
# Epoch 8/10
# 33/33 [==============================] - 70s 2s/step - loss: 0.7919 - acc: 0.9373 - val_loss: 1.3007 - val_acc: 0.8707
# Epoch 9/10
# 33/33 [==============================] - 69s 2s/step - loss: 0.6382 - acc: 0.9412 - val_loss: 3.3948 - val_acc: 0.7586
# Epoch 10/10
# 33/33 [==============================] - 69s 2s/step - loss: 0.3972 - acc: 0.9595 - val_loss: 1.2354 - val_acc: 0.9397