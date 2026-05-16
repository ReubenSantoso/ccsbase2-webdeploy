# CCSBase2

A Collision Cross Section prediction model using a XGBoostRegressor model.

## Dependencies

### 1. Create a Conda environment (Python 3.10)

```bash
conda create -n ccsbase2 python=3.10 -y
conda activate ccsbase2
```

### 2. Install
```bash
pip install numpy pandas scikit-learn rdkit xgboost joblib requests matplotlib
```

## Building Database

### Build Main Database
CCSbase2 aggregates across the 5 different datasets listed below. AllCCS was obtained by downloading chemicals manually using their online database.


#### 
- [CCSBase](https://ccsbase.net/)
- [PNNL](https://pnnl-comp-mass-spec.github.io/MetabolomicsCCS/)
- [ALLCCS](http://allccs.zhulab.cn/database/browser)
- [METLIN-CCS](https://www.dropbox.com/scl/fi/9xctm5ub834muw1qrvd5b/CCS-Publication-V3.zip?e=2&file_subpath=%2FCCS-Publication-V3&rlkey=zi9xaua4zzgpiiaznexabpg7i&dl=0)
- [Dataset found from ACS publication](https://pubs.acs.org/doi/10.1021/acs.jafc.2c00724)


#### [Download Prebuilt Database](https://drive.google.com/file/d/1NQy1ZcuRwRlZv2scIgqvrvsFDLFhEewx/view?usp=sharing)


#### [Download Model Weights](https://drive.google.com/file/d/1sXutsjETBxTs-SutORb6LLbc_sK6KgdY/view?usp=drive_link)


Please note that building the database from scratch takes a very long time as thousands of API calls need to be made. Instructions have been given below.

Run the code below in another file. We advise you to use Google Colab due to PubChem rate limiting IP addresses which results in a "ServerBusy" error. You may need to just call ``ccsml.find_smiles()`` multiple times while commenting out the rest of the method calls.

After determining as many SMILES, call ``ccsml.find_inchikey()`` at the end which calls PubChem API to retrieve InChiKey when given SMILES string. This is to obtain InChiKeys for CCSbase datapoints since CCSbase stores SMILES, but not InChiKeys. 

ClassyFire also does rate limiting, so call ``ccsml.find_inchikey()`` multiple times while commenting out all other method calls. NOTE: APIs will not find SMILES string and/or subclass for all chemicals.

```python
from data import CCSDataIntegration

ccsml = CCSDataIntegration("CCSMLDatabase.db")
ccsml.add_acs()
ccsml.add_ccsbase()
ccsml.add_allccs()
ccsml.add_metlin()
ccsml.add_pnnl()
ccsml.find_smiles()
ccsml.find_inchikey()
ccsml.find_classes()
ccsml.clean()
```


### Classify Unknown Subclasses

ClassyFire will not identify subclasses for all chemicals, so a XGBClassifier was developed to classify the unknown subclasses. All subclasses that are known are grouped and groups less than less than 30 chemicals are not included during training and act as a proxy out of distribution (OOD) dataset.

```python
from classifier import SubclassClassifier

classifier = SubclassClassifier("CCSMLDatabase.db", min_class_count=30, seed=26)
classifier.fit()
classifier.predict()
```


### Build 3D Conformers (Optional)

Abalations have been ran using 3D conformers, however all models we have trained using 3D descriptors perform suboptimally from the current model. This is mostly due to ambiguity with determining the correct conformer that a chemical compound took on while in the mass spectrometry instrument. However, if you would like to try experiments for yourself, run the following code in another file to generate the conformers after you have built the database.


```python
from utils import Utils
import sqlite3

conn = sqlite3.connect("CCSMLDatabase.db")
cur = conn.cursor()

smiles = cur.execute("SELECT DISTINCT smi FROM master_clean").fetchall()

conformers_calculated = cur.execute("SELECT smi FROM conformers").fetchall()
conformer_smiles = set([smi[0] for smi in conformers_calculated])

smiles_reshaped = []
for smi in smiles:
    if smi not in conformer_smiles:
        smiles_reshaped.append(smi)

conformers = Utils().calculate_conformers("CCSMLDatabase.db", smiles_reshaped)
```

## Training Model

Run the following code in another file. Set ``use_metlin=False`` to train without METLIN dataset, and set ``subclass_frequency_threshold=N`` where N > 0 to exclude (subclass,adduct) underrepresented groups from training. You can optionally set ``subclass_frequency_threshold=None`` to include all (subclass,adduct) groups.

```python
from train import CCSBase2

ccs_model = CCSBase2("CCSMLDatabase.db", 
                       "train_data.csv", 
                       "test_data.csv",
                       seed=26,
                       use_metlin=True, 
                       subclass_frequency_threshold=40)
ccs_model.fit()
ccs_model.predict()
```


## Inference

To perform inference, create a .csv file with column names ``smi,adduct`` and ensure saved model is same directory.
Then run the command in your CLI with your csv filename.

```bash
python inference.py ./pretrained/example_inference.csv
```