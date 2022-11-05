# This program predicts binary classicication of Ames Mutagenicity dataset with only MW as descriptors


# import the necessary packages
import random
import numpy as np
import pandas as pd
import scipy

from KiNet_mlp import KiNet_mlp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics




# Getting the list of labels/ targets of molecules
df_target = pd.read_csv('target_mutagenicity_with_shannon.csv', encoding='cp1252') 

# Getting the list of labels/ targets of molecules together with the features: This csv contains already evaluated Shannon entropy values as a feature, along with other features (including MW) and labels.
df =  pd.read_csv('features_mutagenicity_with_shannon.csv', encoding='cp1252') 

# constructing a new df column containng only MW as the sole descriptor and the target labels
df_1 = pd.DataFrame(df.iloc[:,6].values)
df_2 = pd.DataFrame(df.iloc[:,2085].values)
df_new = pd.concat([ df_1, df_2], axis = 1)

    
# Getting the max & min of the target column
maxPrice = df.iloc[:,-1].max() 
minPrice = df.iloc[:,-1].min() 


print("[INFO] constructing training/ testing split")
split = train_test_split(df_new, test_size = 0.15, random_state = 42)  

# split descriptor values between train & test sets
(XtrainTotalData, XtestTotalData) = split  # split format always is in (data_train, data_test, label_train, label_test)
 
# Normalizing the test or Labels
XtrainLabels = (XtrainTotalData.iloc[:,-1])/ (maxPrice)
XtestLabels = (XtestTotalData.iloc[:,-1])/ (maxPrice)  

# Getting the data columns except the last column which is the target column
XtrainData = (XtrainTotalData.iloc[:,0:-2])
XtestData = (XtestTotalData.iloc[:,0:-2])

# perform min-max scaling of each continuous feature column (data columns) in the range [0 1]
cs = MinMaxScaler()
trainContinuous = cs.fit_transform(XtrainTotalData.iloc[:,0:XtrainTotalData.shape[1]-1])
testContinuous = cs.transform(XtestTotalData.iloc[:,0:XtestTotalData.shape[1]-1])

print("[INFO] processing input data after normalization....")
XtrainData, XtestData = trainContinuous,testContinuous


# create the MLP model
mlp = KiNet_mlp.create_mlp(XtrainData.shape[1], regress = False) # the input dimension to mlp would be shape[1] of the matrix i.e. number of column features
combinedInput = mlp.output

# Defining the Final FC (Dense) layers 
x = Dense(100, activation = "relu") (combinedInput)
x = Dense(10, activation = "relu") (combinedInput)
x = Dense(1, activation = "sigmoid") (x)


# FINAL MODEL as 'model'
model = Model(inputs = mlp.input, outputs = x)


# initialize the optimizer model and compile the model
print("[INFO] compiling model...")
opt = SGD(lr= 1.05e-2, decay = 1.05e-2/200)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


# train the network
print("[INFO] training network...")
trainY = XtrainLabels
testY = XtestLabels

# defining some essential hyperparameters: # of epochs & batch size
epoch_number = 150
BS = 50

# Defining the early stop to monitor the validation loss to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=500, verbose=1, mode='auto')

# shuffle = False to reduce randomness and increase reproducibility
H = model.fit( x = XtrainData , y = trainY, validation_data = (XtestData, testY), batch_size = BS, epochs = epoch_number, verbose=1, shuffle=False, callbacks = [early_stop]) 

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(XtestData,batch_size=BS)

pred_val = []
for pred in predictions:
    
    if pred>0.5:
        
        pred_val.append(1)
        
    else:
        pred_val.append(0)
print(classification_report (testY, pred_val, target_names = ["non-mutagenic" , "mutagenic"] ))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch_number ), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch_number ), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoch_number ), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epoch_number ), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Mutagenicity Prediction")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('MLP_only_loss&accuracy')

# forming the confusion matrix
conf_mx = confusion_matrix(testY,pred_val)
plt.matshow(conf_mx, cmap = 'binary')
plt.show()

# evaluating the AUC

# calculate the fpr (false positive) and tpr (true positives) for all thresholds of the classification
preds = predictions
fpr, tpr, threshold = metrics.roc_curve(testY, preds)
roc_auc = metrics.auc(fpr, tpr)

# plotting the ROC_AUC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('MLP_only_AUC_ROC')
plt.show()