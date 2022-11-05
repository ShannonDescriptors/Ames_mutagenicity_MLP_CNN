# This program predicts binary classicication of Ames Mutagenicity dataset with Shannon entropy (SMILES/ SMARTS/InChiKey-based), fractional Shannon entropy, MW and other descriptors


# import the necessary packages
from imutils import paths
import random
import shutil
import os
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

from rdkit import Chem
from rdkit.Chem import AllChem

import re
import math

# K-mer tokenization (optional package)
from SmilesPE.pretokenizer import kmer_tokenizer



# Getting the list of labels/ targets of molecules
df_target = pd.read_csv('target_mutagenicity_with_shannon.csv', encoding='cp1252') 

# Getting the list of labels/ targets of molecules together with the features: This csv contains already evaluated Shannon entropy values as a feature, along with other features (including MW) and labels.
df =  pd.read_csv('features_mutagenicity_with_shannon_with_smiles.csv', encoding='cp1252') 
df_smiles = df.iloc[:,2085].values


# Calculating the shannon entropy for each smiles string using a function definition

# smiles regex definition
SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
regex = re.compile(SMI_REGEX_PATTERN)


def shannon_entropy_smiles(mol_smiles):
    
    molecule = mol_smiles 
    tokens = regex.findall(molecule)
    # tokens = kmer_tokenizer(molecule, ngram=10)
    
    ### Frequency of each token generated
    L = len(tokens)
    L_copy = L
    tokens_copy = tokens
    
    num_token = []
    
    
    for i in range(0,L_copy):
        
        token_search = tokens_copy[0]
        num_token_search = 0
        
        if len(tokens_copy) > 0:
            for j in range(0,L_copy):
                if token_search == tokens_copy[j]:
                    # print(token_search)
                    num_token_search += 1
            # print(tokens_copy)        
                    
            num_token.append(num_token_search)   
                
            while token_search in tokens_copy:
                    
                tokens_copy.remove(token_search)
                    
            L_copy = L_copy - num_token_search
            
            if L_copy == 0:
                break
        else:
            pass
        
    # print(num_token)
    
    ### Calculation of Shannon entropy
    total_tokens = sum(num_token)
    
    import math
    shannon = 0
    
    for k in range(0,len(num_token)):
        
        pi = num_token[k]/total_tokens
        
        # print(num_token[k])
        # print(math.log2(pi))
        
        shannon = shannon - pi * math.log2(pi)
    
    # shannon = math.exp(-shannon)    
        
    return shannon    

# Constructing the padded array of partial shannon per molecule
def ps_padding(ps, max_len_smiles):
    
    len_ps = len(ps)
    
    len_forward_padding = int((max_len_smiles - len_ps)/2)
    len_back_padding = max_len_smiles - len_forward_padding - len_ps
    
    ps_padded = list(np.zeros(len_forward_padding))  + list(ps) + list(np.zeros(len_back_padding))
    
    return ps_padded 


# generating a dictionary of atom occurrence frequencies
def freq_atom_list(atom_list_input_mol):
    
    atom_list = ['H', 'B', 'C', 'Si', 'N', 'P', 'O', 'S', 'F', 'Cl', 'Se', 'Br', 'I'] 
    dict_freq = {}
    
    ### adding keys
    for i in range(len(atom_list)):
        dict_freq[atom_list[i]] = 0  ### The values are all set 0 initially
    # print(dict_freq)
    
    ### update the value by 1 when a key in encountered in the string
    for i in range(len(atom_list_input_mol)):
        dict_freq[ atom_list_input_mol[i] ] = dict_freq[ atom_list_input_mol[i] ] + 1
    
    ### The dictionary values as frequency array
    freq_atom_list =  list(dict_freq.values())/ (  sum(  np.asarray (list(dict_freq.values()))  )    )
    
    # print(list(dict_freq.values()))
    # print(freq_atom_list )
    
    ### Getting the final frequency dictionary
    ### adding values to keys
    for i in range(len(atom_list)):
        dict_freq[atom_list[i]] = freq_atom_list[i]  
        
    # print(dict_freq)
    freq_atom_list = dict_freq
    
    return freq_atom_list


# estimating the max length of smiles strings
len_smiles = []
for j in range(0,len(df_smiles)):
    
    mol = Chem.MolFromSmiles(df_smiles[j])
    mol = Chem.AddHs(mol)
    
    k=0
    for atom in mol.GetAtoms():
        k = k +1 
    len_smiles.append(k)

    
max_len_smiles = max(len_smiles)


# Collecting the fingerprints to use as descriptor array
shannon_arr = []
fp_combined = []
MW_list = []

for i in range(0,len(df_smiles)):  
    
    
  mol_smiles = df_smiles[i]
  mol = Chem.MolFromSmiles(mol_smiles)
  mol = Chem.AddHs(mol)
  
  
  # estimating the partial shannon for an atom type => the current node
  total_shannon = shannon_entropy_smiles(mol_smiles)
  # shannon_arr.append( total_shannon )
  
  # The atom list as per rdkit in string form
  atom_list_input_mol = []
  for atom_rdkit in mol.GetAtoms():
     atom_list_input_mol.append(str(atom_rdkit.GetSymbol()))     
        
     
  freq_list_input_mol = freq_atom_list(atom_list_input_mol)
  
  # estimating the fractional Shannon or partial shannon entropies
  ps = []
  for atom_rdkit in mol.GetAtoms():
      atom_symbol = atom_rdkit.GetSymbol()
      atom_type = atom_symbol ### atom symbol in atom type
      
      partial_shannon = freq_list_input_mol[atom_type] * total_shannon
      ps.append( freq_list_input_mol[atom_type] )
  

  ps_arr = ps_padding(ps, max_len_smiles)     
  fp_combined.append(ps_arr)


# partial shannon_entropy as feature
fp_mol = pd.DataFrame(fp_combined)


# constructing a new df dataframe containng all descriptors estimated, shannon values, partial shannon and labels
df_1 = pd.DataFrame(df.iloc[:,0:2084].values)
df_2 = pd.DataFrame(df.iloc[:,2084].values)
df_3 = pd.DataFrame(df.iloc[:,2086].values)


# df_new = pd.concat([ df_1, df_2, df_3 ], axis = 1)
df_new = pd.concat([ df_1, df_2, fp_mol, df_3 ], axis = 1)

    
# Getting the max & min of the target column
maxPrice = df_new.iloc[:,-1].max() # grab the maximum val in the training set's last column
minPrice = df_new.iloc[:,-1].min() # grab the minimum val in the training set's last column


print("[INFO] constructing training/ testing split")
split = train_test_split(df_new, test_size = 0.15, random_state = 10) 

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


# Defining the Final FC layers (Dense) towards regression
x = Dense(100, activation = "relu") (combinedInput)
x = Dense(10, activation = "relu") (combinedInput)
x = Dense(1, activation = "sigmoid") (x)


# FINAL MODEL as 'model'
model = Model(inputs = mlp.input, outputs = x)


## initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = SGD(lr= 1.05e-2, decay = 1.05e-2/200)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
trainY = XtrainLabels
testY = XtestLabels

epoch_number = 150
BS = 50


# Defining the early stop to monitor the validation loss to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=500, verbose=1, mode='auto')

# shuffle = False to reduce randomness and increase reproducibility
H = model.fit( x = XtrainData , y = trainY, validation_data = ( XtestData, testY), batch_size = BS, epochs = epoch_number, verbose=1, shuffle=False, callbacks = [early_stop]) 

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

# Calculating the actual F1-score
from sklearn.metrics import f1_score
print("F1-score of the classifier: ", f1_score(testY, pred_val))

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
plt.savefig('loss&accuracy_with_all_descriptors_partial_shannon_MLP_only')

# evaluating the confusion matrix
conf_mx = confusion_matrix(testY,pred_val)
plt.matshow(conf_mx, cmap = 'binary')
plt.show()

# Estimating the AUC
# calculate the fpr and tpr for all thresholds of the classification
preds = predictions
fpr, tpr, threshold = metrics.roc_curve(testY, preds)
roc_auc = metrics.auc(fpr, tpr)

# plotting the ROC_AUC curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('AUC_ROC_with_all_descriptors_partial_shannon_MLP_only')
plt.show()