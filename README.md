# Ames_mutagenicity_MLP_CNN

Harnessing Shannon entropy of molecular symbols in deep neural networks to enhance prediction accuracy
------------------------------------------------------------------------------------------------------
This repository holds the codes pertaining to Fig. 2c of the article 'Harnessing Shannon entropy of molecular symbols in deep neural networks to enhance prediction accuracy'.

Description
-----------
Shannon entropy framework has been demonstrated as an efficient descriptor for classification-type machine learning problem using (i) MLP-based and (ii) MLP+CNN based-deep neural networks. In this specific case, we model or classify the toxicity labels as per the Ames mutagenicity data set. The specific objectives of the codes are described in the Notes section below. The basic dataset has been provided in the repository in the form of .csv files within the 'features_mutagenicity_with_shannon_with_smiles.rar' file.

Usage
-----
1. Download or make a clone of the repository. Unzip the features_mutagenicity_with_shannon_with_smiles.rar to be used as a .csv data file
2. Make a new conda environment using the environment file 'mlp_dnn.yml'
3. Run the python files directly using a python IDE or from command line

Example: python MLP_only_train_test_with_shannon_partial_shannon_smiles_inchikey.py

Notes
-----
  The function files are KiNet_mlp.py and image_and_table_processing.py. Therefore, directly run the other python files apart from these. 

  The objectives and usage of the rest of the scripts are as follows: Please run the python scripts directly or using the command line 'python <script_name.py> from the terminal.

  (i) Image dataset download and data acquisition: Run the chembl_target_featurizer_Ki_with_shannon_mod_wo_H_with_smiles.py file directly to build the image dataset which will be saved in the folder target_images_mutagenicity_with_shannon_wo_H. This script also extracts and saves a descriptor set from the CHEMBL website to features_mutagenicity_with_shannon_with_smiles.csv which would be used in all other scripts as the dataset file.

  (ii) MLP_only_train_test_hybrid_without_shannon.py: This script models binary classification of toxicity data as per Ames mutagenicity dataset using MW as descriptor.  The model predicts binary classification of toxicity of molecules as per the test data set.

  (iii) MLP_only_train_test_with_shannon.py:This script models and predicts binary classicication of Ames Mutagenicity dataset with Shannon entropy and MW as descriptors. 

  (iv) MLP_only_train_test_with_shannon_partial_shannon_smiles_inchikey.py: This program build model and predicts binary classicication of Ames Mutagenicity dataset with Shannon entropy (SMILES/ SMARTS/InChiKey-based), fractional Shannon entropy, bond (type) frequency and MW as descriptors.

  (v) MLP_only_train_test_hybrid_with_partial_shannon_all_descriptors.py: This script build model and predicts binary classicication of Ames Mutagenicity dataset with Shannon entropy (SMILES/ SMARTS/InChiKey-based), fractional Shannon entropy, MW and other descriptors as obtained from runnning the script mentioned in (i).

  (vi) CNN_MLP_train_test_hybrid_with_partial_shannon_all_descriptors.py: This script builds model and predicts binary classicication of Ames Mutagenicity dataset with Shannon entropy (SMILES/ SMARTS/InChiKey-based), fractional Shannon entropy, MW and other descriptors using a hybrid MLP and 2D image dataset-based CNN model.


