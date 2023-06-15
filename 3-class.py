
!pip install scikit-learn
!pip install pydot
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation, LSTM, Dropout, Flatten, Concatenate, Conv1D, Input, MaxPooling1D, AveragePooling1D, BatchNormalization, GlobalAveragePooling1D, concatenate
from tensorflow.keras import Model
import pandas as pd
import matplotlib.pyplot as plt
#import random
#np.random.seed(0)
#random.seed.value
#random.seed(0)

df=pd.read_csv('Epileptic Seizure Recognition.csv')
dic = {5: 0, 4: 0, 3: 2, 2: 2, 1: 1}
df['y'] = df['y'].map(dic)

df.head()

X=df.values

X=X[:,1:-1].astype(np.float32)

y=np.array(df['y']).astype(np.float32)
Y=np_utils.to_categorical(y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=1)

print(X_train.shape,Y_train.shape)

def dense_block(neurons, input_shape):
    k = tf.keras.layers.BatchNormalization() (input_shape)
    k = tf.keras.layers.Dense(neurons, activation=tf.keras.layers.LeakyReLU(0.2)) (k)
    k = tf.keras.layers.Dropout(0.4) (k)
    k = tf.keras.layers.Dense(neurons, activation=tf.keras.layers.LeakyReLU(0.2)) (k)
    k = tf.keras.layers.Dropout(0.4) (k)
    k = tf.keras.layers.Dense(neurons, activation=tf.keras.layers.LeakyReLU(0.2)) (k)
    k = tf.keras.layers.Dropout(0.4) (k)
    k = tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(0.2)) (k)
    k = tf.keras.layers.Dropout(0.4) (k)
    k = tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(0.2)) (k)
    return k

input = tf.keras.layers.Input(shape=(178, 1), name='input')
m1 = dense_block(256, input)
m2 = dense_block(512, input)
m3 = dense_block(1024, input)
m4 = dense_block(256, input)
m5 = dense_block(512, input)
m6 = dense_block(1024, input)

z = tf.keras.layers.Concatenate()([m1, m2, m3, m4, m5, m6])
z = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(0.1)) (z)
z = tf.keras.layers.BatchNormalization() (z)
z = tf.keras.layers.Dropout(0.3) (z)
output = tf.keras.layers.Dense(3, activation='softmax',trainable=True, name='output') (z)
model1 = tf.keras.models.Model(input, output)

model1.summary()

from keras.utils import plot_model
plot_model(model1, to_file='model.png')

#seed=[0,100,200,300]
#acc=[]
#for i in seed:
 #   np.random.seed(i)
#random.seed.value
  #  random.seed(i)'''

model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),loss='categorical_crossentropy',metrics=['accuracy'])

checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    min_delta=0.0008,
    mode='min'
)

history=model1.fit(X_train,Y_train,700,500,verbose=1,validation_data=(X_test,Y_test), callbacks=[model_checkpoint_callback])

ypred=model1.predict(X_test)
#ypred.shape()

yp=np.zeros((Y_test.shape[0]))
yo=np.ones((Y_test.shape[0]))

for i in range(Y_test.shape[0]):
    yp[i]=np.argmax(ypred[i])
    yo[i]=np.argmax(Y_test[i])

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
accuracy_score(yo,yp)

from sklearn.metrics import precision_score
precision_score(yo, yp, labels=[1,2,3], average='micro')

from sklearn.metrics import recall_score
recall_score(yo, yp, labels=[1,2,3], average='micro')

from sklearn.metrics import f1_score
f1_score(yo, yp, labels=[1,2,3], average='micro')

cm=confusion_matrix(yo,yp)
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['1','2','3'],
            yticklabels=['1','2','3'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.savefig('3-class-confusion.png', dpi=500)
plt.show()

print(classification_report(yo,yp))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('acc-plt-3cl.png', dpi=500)
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss-plt-3cl.png', dpi=500)
plt.show()

model1.save('Models/Ternary model.h5')