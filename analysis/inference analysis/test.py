from rdkit import Chem
from rdkit.Chem import Draw, rdFingerprintGenerator

def visualize_morgan_bit(mols, bit_to_draw, radius=2, n_bits=1024):
    """
    Finds a molecule containing 'bit_to_draw' and visualizes that specific substructure.
    """
    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=True,
        countSimulation=True,
    )

    for mol in mols:
        if mol is None:
            continue

        # bit_info stores {bit_id: ((atom_idx, radius), ...)}
        additional_output = rdFingerprintGenerator.AdditionalOutput()
        additional_output.AllocateBitInfoMap()
        _ = fpgen.GetCountFingerprint(mol, additionalOutput=additional_output)
        bit_info = additional_output.GetBitInfoMap()

        if bit_to_draw in bit_info:
            # Generate the image for the first instance of this bit in the molecule
            print(f"Found bit {bit_to_draw} in molecule: {Chem.MolToSmiles(mol)}")
            return Draw.DrawMorganBit(mol, bit_to_draw, bit_info)

    print(f"Bit {bit_to_draw} not found in the provided molecule list.")
    return None

# --- Example Usage ---
# 1. Load your molecules (replace with your actual dataset)
smiles_list = ['Cc1ccccc1', 'CN(C)C(=O)c1ccc(cc1)NC(=O)C', 'CCO']
mols = [Chem.MolFromSmiles(s) for s in smiles_list]

# 2. Visualize a specific bit from your importance chart (e.g., 531)
# Note: Ensure radius and n_bits match your model's training parameters.
# In this project, train.py uses radius=2 and fpSize=1024.
img = visualize_morgan_bit(mols, bit_to_draw=531, radius=2, n_bits=1024)

# To display in Jupyter:
# display(img)
