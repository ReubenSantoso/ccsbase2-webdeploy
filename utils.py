import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split

from rdkit import Chem

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator, AllChem

class Utils:
    # ============= CCS Prediction ==============
    def train_test_split_custom(self,
        database_file,
        train_csv_path,
        test_csv_path,
        test_size=0.2,
        random_state=26,
        use_metlin=True,
        subclass_frequency_threshold=None
    ):
        conn = sqlite3.connect(database_file)
        query = "SELECT smi, mass, z, ccs, name, subclass, adduct, tag FROM master_clean WHERE ABS(z) = 1 AND subclass != 'NONE (predicted)'"
        if not use_metlin:
            query += " AND tag != 'METLIN'"

        df = pd.read_sql_query(query, conn)
        conn.close()

        # Remove the (predicted) tag from predicted classes 
        df["subclass"] = df["subclass"].str.replace(r" \(predicted\)$", "", regex=True)

        df["count"] = df.groupby(["subclass", "adduct"])["subclass"].transform("count")
        df_split = df
        if subclass_frequency_threshold:
            df_split = df[df['count'] >= subclass_frequency_threshold]

        train_parts = []
        test_parts = []

        for (_, _), group_df in df_split.groupby(["subclass", "adduct"]):
            group_df = group_df.copy()
            if len(group_df) < 2:
                train_parts.append(group_df)
                continue

            y = group_df["ccs"].values
            stratify_labels = None

            y_arr = y.astype(float)
            stratify_labels = (y_arr // 10).astype(int)
            vc = pd.Series(stratify_labels).value_counts()
            if vc.min() < 2:
                stratify_labels = None
            else:
                n_samples = len(y_arr)
                n_test = int(np.ceil(test_size * n_samples))
                n_classes = vc.size
                if n_test < n_classes:
                    stratify_labels = None

            group_train, group_test = sk_train_test_split(
                group_df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_labels,
            )

            train_parts.append(group_train)
            test_parts.append(group_test)

        if train_parts:
            train_df = pd.concat(train_parts, ignore_index=True)
        else:
            train_df = pd.DataFrame(columns=df.columns)

        if test_parts:
            test_df = pd.concat(test_parts, ignore_index=True)
        else:
            test_df = pd.DataFrame(columns=df.columns)

        if "count" in train_df.columns:
            train_df = train_df.drop(columns=["count"])
        if "count" in test_df.columns:
            test_df = test_df.drop(columns=["count"])

        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)

        print(len(train_df), "train rows")
        print(len(test_df), "test rows")

    def calculate_descriptors(self, smiles: str, ion_mass: float, charge: int, adducts: list, adduct: str):
        """
        Calculate molecular descriptors for CCS prediction.

        Parameters
        ----------
        smiles : str
            SMILES string of the molecule.
        ion_mass : float
            Observed ion mass (m/z * charge).
        charge : int
            Ion charge state (z).
        adducts : list
            List of known adduct types (sorted, with count >= 100 in training data).
        adduct : str
            Adduct type for this molecule.

        Returns
        -------
        np.ndarray
            Feature vector with the following structure:

            Index       | Feature Name      | Description
            ------------|-------------------|------------------------------------------
            0           | MolecularWeight   | Exact molecular weight (RDKit CalcExactMolWt)
            1           | AdductMass        | ion_mass - MolecularWeight
            2           | Charge            | Ion charge state (z)
            3           | LabuteASA         | Labute accessible surface area
            4 to 4+N    | Adduct_<name>     | One-hot encoding for each adduct in list (N = len(adducts))
            4+N         | Adduct_Other      | One-hot flag for unknown adducts
            4+N+1 to end| MorganFP_0..1023  | Morgan count fingerprint (radius=2, 1024 bits, with chirality)

            Total length: 4 + len(adducts) + 1 + 1024
        """
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        feature_values = []

        molecular_weight = rdMolDescriptors.CalcExactMolWt(mol)
        feature_values.append(molecular_weight)
        feature_values.append(ion_mass - molecular_weight)   # AdductMass
        feature_values.append(charge)
        feature_values.append(rdMolDescriptors.CalcLabuteASA(mol))
        
        ohe_adduct = [0] * (len(adducts) + 1)
        ohe_index = len(adducts)
        if adduct in adducts:
            ohe_index = adducts.index(adduct)
        ohe_adduct[ohe_index] = 1

        feature_values.extend(ohe_adduct)

        # Morgan count fingerprint
        morgan_count_fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=1024, includeChirality=True, countSimulation=True
        )
        count_fp = morgan_count_fpgen.GetCountFingerprint(mol)
        arr = np.zeros((count_fp.GetLength(),), dtype=int)
        DataStructs.ConvertToNumpyArray(count_fp, arr)
        feature_values.extend(arr.tolist())

        return np.array(feature_values)
    
    # NOT USED
    def calculate_3d_descriptors(self, smiles: str, ion_mass: float, charge: int, database_file: str):
        conn = sqlite3.connect(database_file)
        cur = conn.cursor()

        molblob = cur.execute("SELECT mol_blob FROM conformers WHERE smi = ?", (smiles,)).fetchone()

        if not molblob:
            return None
        
        # Rehydrate RDKit molecule (includes conformer)
        mol = Chem.Mol(molblob[0])

        # Ensure we actually have a conformer
        if mol.GetNumConformers() == 0:
            return None

        feature_values = []

        mw = rdMolDescriptors.CalcExactMolWt(mol)
        feature_values.append(mw)                          # MolecularWeight
        feature_values.append(ion_mass - mw)               # AdductMass
        feature_values.append(charge)                      # Charge

        dclv = rdMolDescriptors.DoubleCubicLatticeVolume(mol)
        feature_values.append(dclv.GetVDWVolume())
        feature_values.append(dclv.GetSurfaceArea())

        feature_values.append(rdMolDescriptors.CalcLabuteASA(mol))

        # Morgan count fingerprint
        morgan_count_fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=1024, includeChirality=True, countSimulation=True
        )
        count_fp = morgan_count_fpgen.GetCountFingerprint(mol)
        arr = np.zeros((count_fp.GetLength(),), dtype=int)
        DataStructs.ConvertToNumpyArray(count_fp, arr)
        feature_values.extend(arr.tolist())

        return np.array(feature_values)
    
    # =========== 3D Conformers =============

    def calculate_conformers(self, database_file: str, smiles: list):
        conn = sqlite3.connect(database_file)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS conformers (
                smi TEXT PRIMARY KEY,
                mol_blob BLOB NOT NULL
            )
        """)

        count = 0
        for smi in smiles:
            base = Chem.MolFromSmiles(smi)
            if base is None:
                continue

            mol = Chem.AddHs(base)

            params = AllChem.ETKDGv3()
            params.randomSeed = 26
            conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=50, params=params))
            if not conf_ids:
                continue

            results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)

            energies = [
                (cid, res[1])
                for cid, res in zip(conf_ids, results)
                if res[0] == 0  # strict success only
            ]
            if not energies:
                continue

            best_conf_id, best_energy = min(energies, key=lambda x: x[1])

            # Copy molecule first so original conformers remain valid
            best_mol = Chem.Mol(mol)

            # Deep-copy the conformer from the original molecule
            best_conf = Chem.Conformer(mol.GetConformer(best_conf_id))

            # Keep only that conformer on best_mol
            best_mol.RemoveAllConformers()
            best_mol.AddConformer(best_conf, assignId=True)  # confId becomes 0

            mol_blob = best_mol.ToBinary()

            cur.execute(
                "INSERT INTO conformers (smi, mol_blob) VALUES (?, ?)",
                (smi, sqlite3.Binary(mol_blob))
            )

            count += 1
            if count % 100 == 0:
                conn.commit()


        conn.commit()
        conn.close()
            
    # =========== Classifier ============

    def calculate_classifier_descriptors(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        morgan_count_fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=1024, includeChirality=True, countSimulation=True
        )

        count_fp = morgan_count_fpgen.GetCountFingerprint(mol)
        arr = np.zeros((count_fp.GetLength(),), dtype=int)
        DataStructs.ConvertToNumpyArray(count_fp, arr)

        return arr.astype(np.float32)
