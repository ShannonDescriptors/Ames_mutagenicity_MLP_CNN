# This program evaluates a set of descriptors of molecules using smiles strings

# import necessary packages 
import pandas as pd
import numpy as np

# Import necessary packages from rdkit for constructing images from smiles
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors as desc


from rdkit.Chem.Draw import MolToImage, MolDrawOptions
opts = MolDrawOptions()

# Importing MHFP fingerprinting package
from mhfp.encoder import MHFPEncoder

import re
import math

# K-mer tokenization (optional package)
from SmilesPE.pretokenizer import kmer_tokenizer

# Getting the list of molecules from the following .xls file 
target_data_list = pd.read_excel('Mutagenicity_N6512.xls') 
mol_list = target_data_list['Canonical_Smiles'].values

# Extracting the smiles and activity (target) values
df_SMILES = mol_list
df_Activity = target_data_list['Activity'].values


# ####--------------------------------------------Keep this section commented if not constructing 2D image dataset from smiles: only for image generation purpose-------------------------------------------------
# ### Making the folder 'target_images', for keeping the images, in the current directory if not already created
# import os
# if not os.path.isdir('target_images_mutagenicity'):
#     os.makedirs('target_images_mutagenicity')

# ### loop over to get the SMILES strings and corresponding images saved as .png in target_images folder
# mutagenicity = []
# n = len(df_SMILES)
# for i in range(n):
    
#     try:
#         mol = Chem.MolFromSmiles(df_SMILES[i])
#         ##mol = Chem.AddHs(mol) 
    
#     ### MolToImage not working in the current version of rdkit
#     # for atom in mol.GetAtoms():
#     #     j = atom.GetIdx()
#     #     opts.atomLabels[j] = ''  ## '' getting rid of atom labels

#     #     img = MolToImage(mol, options=opts)
    
#     # img.save('target_images\m_{}.png'.format(i))
    
#     ### saving images with atomLabels
#         Draw.MolToFile(mol,'target_images_mutagenicity\m_{}.png'.format(i), drawingOptions=opts)
#         # print("Processed {}-th image".format(i))
        
#         mutagenicity.append(df_Activity[i])
        
#     except:
#         print("Failed for SMILES {}".format(df_SMILES[i]))

# df_target = pd.DataFrame(mutagenicity)
# df_target.to_csv('target_mutagenicity.csv', index=False)

# ###------------------------------------------Keep this section commented if not constructing 2D image dataset from smiles: only for older rdkit version/ images w/o labels & with Hs----------------------------------------

# ### Making the folder 'target_images', for keeping the images, in the current directory if not already created
# import os
# if not os.path.isdir('target_images_mutagenicity_with_shannon_wo_H'):
#     os.makedirs('target_images_mutagenicity_with_shannon_wo_H')

# ### loop over to get the SMILES strings and corresponding images saved as .png in target_images folder
# mutagenicity = []
# n = len(df_SMILES)
# n = 50
# for i in range(n):
    
#     try:
#         mol = Chem.MolFromSmiles(df_SMILES[i])
#         # mol = Chem.AddHs(mol) 
    
#     ## MolToImage not working in the current version of rdkit
#         for atom in mol.GetAtoms():
#             j = atom.GetIdx()
#             opts.atomLabels[j] = ''  ## '' getting rid of atom labels

#             img = MolToImage(mol, options=opts)
    
#             img.save('target_images_mutagenicity_with_shannon_wo_H\m_{}.png'.format(i))
    
#         # ### saving images with atomLabels
#         # Draw.MolToFile(mol,'target_images_mutagenicity\m_{}.png'.format(i), drawingOptions=opts)
#         # # print("Processed {}-th image".format(i))
        
#         mutagenicity.append(df_Activity[i])
        
#     except:
#         print("Failed for SMILES {}".format(df_SMILES[i]))

# df_target = pd.DataFrame(mutagenicity)
# df_target.to_csv('target_mutagenicity_with_shannon.csv', index=False)

# ####-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Function definitions for evaluating descriptors from SMILES strings using rdkit package

def pharma_rules_filter(molecular_weight,logp,h_bond_donor,h_bond_acceptors,rotatable_bonds,number_of_atoms,molar_refractivity,topological_surface_area_mapping,formal_charge,heavy_atoms,num_of_rings):
    
    ### Fail =>0  pass=> 1
    Lipinski_Rule_of_5 = 0
    Ghose_Filter = 0
    Veber_Filter = 0
    Rule_of_3_Filter = 0
    REOS_Filter = 0
    Drug_like_Filter = 0
    
    arr = [Lipinski_Rule_of_5, Ghose_Filter, Veber_Filter, Rule_of_3_Filter, REOS_Filter, Drug_like_Filter ]
    
    # Lipinski
    if molecular_weight <= 500 and logp <= 5 and h_bond_donor <= 5 and h_bond_acceptors <= 10 and rotatable_bonds <= 5:
        arr[0] = 1
        
    # Ghosh
    if molecular_weight >= 160 and molecular_weight <= 480 and logp >= -0.4 and logp <= 5.6 and number_of_atoms >= 20 and number_of_atoms <= 70 and molar_refractivity >= 40 and molar_refractivity <= 130:
        arr[1] = 1
        
    # Veber
    if rotatable_bonds <= 10 and topological_surface_area_mapping <= 140:
        arr[2] = 1
    
    # Rule of 3
    if molecular_weight <= 300 and logp <= 3 and h_bond_donor <= 3 and h_bond_acceptors <= 3 and rotatable_bonds <= 3:
        arr[3] = 1
    
    # REOS
    if molecular_weight >= 200 and molecular_weight <= 500 and logp >= int(0 - 5) and logp <= 5 and h_bond_donor >= 0 and h_bond_donor <= 5 and h_bond_acceptors >= 0 and h_bond_acceptors <= 10 and formal_charge >= int(0-2) and formal_charge <= 2 and rotatable_bonds >= 0 and rotatable_bonds <= 8 and heavy_atoms >= 15 and heavy_atoms <= 50:    
        arr[4] = 1
    
    # Drug like (qed)
    if molecular_weight < 400 and num_of_rings > 0 and rotatable_bonds < 5 and h_bond_donor <= 5 and h_bond_acceptors <= 10 and logp < 5:
        arr[5] = 1
    
    
    return arr

# Calculate functional (rule_arr) projection on structural part (fp or molecular fingerprint array) 
def projection(rule_arr , fp):
    
    n = len(rule_arr)
    
    m = len(fp)
    
    agg = []
    
    for k in range (n):
        
        sum = 0
        for p in range (m):
            
            sum = sum + rule_arr[k] * fp[p]
            
        agg.append(sum)
        
    return agg    


### logP, MSA, HBA, HBD estimation using rdkit.Chem.QED.properties(mol)
# MW = []
MHFP = []
Lip = []
qed_prop = []
structure_function_conv = []

feat = []
smiles_actual = []
mutagenicity = []


# defining the smiles regex
SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
regex = re.compile(SMI_REGEX_PATTERN)

shannon_arr = []
n = len(df_SMILES)

for i in range(n):
    
    try:
        mol = Chem.MolFromSmiles(df_SMILES[i])
        
        mol_smiles = df_SMILES[i]
        ## SECFP (SMILES Extended Connectifity Fingerprint)
        fp = MHFPEncoder.secfp_from_smiles(in_smiles = mol_smiles, length=2048, radius=3, rings=True, kekulize=True, sanitize=False)
        MHFP.append(fp)
        
        # mol = Chem.RemoveHs(mol)
        mol = Chem.AddHs(mol)
        
        ###--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        molecular_weight = desc.ExactMolWt(mol)
        logp = desc.MolLogP(mol)
        h_bond_donor = desc.NumHDonors(mol)
        h_bond_acceptors = desc.NumHAcceptors(mol)
        rotatable_bonds = desc.NumRotatableBonds(mol)
        number_of_atoms = Chem.rdchem.Mol.GetNumAtoms(mol)
        molar_refractivity = Chem.Crippen.MolMR(mol)
        topological_surface_area_mapping = Chem.QED.properties(mol).PSA
        formal_charge = Chem.rdmolops.GetFormalCharge(mol)
        heavy_atoms = Chem.rdchem.Mol.GetNumHeavyAtoms(mol)
        num_of_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
        
        rule_arr = pharma_rules_filter(molecular_weight,logp,h_bond_donor,h_bond_acceptors,rotatable_bonds,number_of_atoms,molar_refractivity,topological_surface_area_mapping,formal_charge,heavy_atoms,num_of_rings)
        
        # fp_conv = np.convolve( fp, rule_arr, mode = 'full')
        # structure_function_conv.append(fp_conv)
        
        # projection contribution between rule_arr (function) & fp (structure)
        structure_function_conv.append(projection(rule_arr , fp))
        
        ###------------------------------------------------------------------------------------------------------------------------------
        
        qed_prop.append(np.asarray( QED.properties(mol)[0:]  ) )
        # MW.append( desc.MolWt(mol) )
        
        Lip_encoding = [Lipinski.FractionCSP3(mol),Lipinski.NOCount(mol),Lipinski.NumAliphaticRings(mol),
                        Lipinski.NumHDonors(mol),Lipinski.NumHAcceptors(mol),Lipinski.NumRotatableBonds(mol),
                        Lipinski.NumRotatableBonds(mol),Lipinski.NumHeteroatoms(mol),Lipinski.HeavyAtomCount(mol),
                        Lipinski.RingCount(mol),Lipinski.NHOHCount(mol),Lipinski.NumAliphaticCarbocycles(mol),
                        Lipinski.NumAliphaticHeterocycles(mol),Lipinski.NumAromaticCarbocycles(mol),Lipinski.NumAromaticHeterocycles(mol),
                        Lipinski.NumAromaticRings(mol)]
        
        Lip.append( Lip_encoding )
        
        ###-------------------------------------------------------------------------Shannon entropy estimation------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        molecule = df_SMILES[i]
        tokens = regex.findall(molecule)
        # tokens = kmer_tokenizer(molecule, ngram=1)
        
        ### Frequency of each token generated
        L = len(tokens)
        L_copy = L
        tokens_copy = tokens
        
        num_token = []
        
        
        for j in range(0,L_copy):
            
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
        
        shannon = 0
        
        for k in range(0,len(num_token)):
            
            pi = num_token[k]/total_tokens
            
            # print(num_token[k])
            # print(math.log2(pi))
            
            shannon = shannon - pi * math.log2(pi)    
            
        shannon_arr.append(shannon)   
        
        smiles_actual.append(df_SMILES[i])
        mutagenicity.append(df_Activity[i])
        desc_combined = list(rule_arr) + list(QED.properties(mol)[0:]) + list(Lip_encoding) + list(projection(rule_arr , fp)) + list(fp) ### This is a row-wise array/ list
        feat.append(desc_combined)

    except:
        print("Failed for SMILES {}".format(df_SMILES[i])) 
    

df_target = pd.DataFrame(mutagenicity)
df_target.to_csv('target_mutagenicity_with_shannon.csv', index=False)        


df_feature = pd.DataFrame(feat)
df_shannon = pd.DataFrame(shannon_arr)
df_smiles = pd.DataFrame(smiles_actual)

# Concatenate the target column to the df_feature, df_shannon and df_smiles columns: saving the csv for using it in modeling
df_feature_target = pd.concat([df_feature, df_shannon, df_smiles, df_target], axis=1)
df_feature_target.to_csv('features_mutagenicity_with_shannon_with_smiles.csv', index=False)