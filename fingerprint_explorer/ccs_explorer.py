import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import DataStructs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import shap
import joblib
import xgboost as xgb
from PIL import Image

st.set_page_config(layout="wide", page_title="CCS Fingerprint Explorer", page_icon="🧪")

# ── Config ─────────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent


def _resolve_existing_file(description: str, candidates: tuple[Path, ...]) -> str:
    """Pick first existing path (Streamlit Cloud cwd varies; assets may live in several layouts)."""
    for path in candidates:
        if path.is_file():
            return str(path)
    lines = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"{description} not found. Checked:\n  {lines}\n"
        "Add ccsbase2.joblib and CCSMLDatabase.db to the deployed repo "
        "(README download links). If they are listed in .gitignore, remove those "
        "entries or place copies Streamlit can clone."
    )


DB_PATH = _resolve_existing_file(
    "CCSMLDatabase.db",
    (
        _REPO_ROOT / "ccsbase2" / "CCSMLDatabase.db",
        _REPO_ROOT / "datasets" / "CCSMLDatabase.db",
        _REPO_ROOT / "CCSMLDatabase.db",
        _APP_DIR / "CCSMLDatabase.db",
    ),
)
MODEL_PATH = _resolve_existing_file(
    "ccsbase2.joblib",
    (
        _REPO_ROOT / "ccsbase2" / "ccsbase2.joblib",
        _REPO_ROOT / "ccsbase2.joblib",
        _APP_DIR / "ccsbase2.joblib",
    ),
)
IMG_W, IMG_H = 400, 350
WATERFALL_W  = 500
PAGE_SIZE    = 5

BIT_COLORS = [
    (0.95, 0.25, 0.25),
    (0.20, 0.55, 0.95),
    (0.15, 0.75, 0.30),
    (0.95, 0.55, 0.05),
    (0.70, 0.15, 0.85),
]
BIT_COLOR_HEX = ["#F24040", "#3399F2", "#27BF4D", "#F28C0D", "#B326D9"]

# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_subclass_counts():
    con = sqlite3.connect(DB_PATH)
    df_counts = pd.read_sql(
        "SELECT subclass, COUNT(*) as cnt FROM master_clean GROUP BY subclass ORDER BY cnt DESC",
        con
    )
    adducts_df = pd.read_sql(
        "SELECT adduct FROM master_clean GROUP BY adduct HAVING COUNT(*) >= 100 ORDER BY adduct",
        con
    )
    con.close()
    df_counts = df_counts[
        ~df_counts["subclass"].str.contains("(predicted)", regex=False)
    ].reset_index(drop=True)
    adducts = [a[0] for a in sorted(adducts_df.to_numpy().tolist())]
    return df_counts, adducts

@st.cache_data
def load_subclass_data(subclass):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT smi, subclass, ccs, mass, z, adduct FROM master_clean WHERE subclass = ?",
        con, params=(subclass,)
    )
    con.close()
    return df

# ── Fingerprint generators ─────────────────────────────────────────────────────
morgan_count_fpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=1024, includeChirality=True, countSimulation=True
)
morgan_bit_fpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=1024, includeChirality=True
)

def sanitize(name):
    return (name.replace("[", "").replace("]", "").replace("<", "")
                .replace("+", "plus").replace("-", "minus"))

def featurise_all(df_sub, ADDUCTS):
    """Featurise every molecule. Returns parallel lists + valid-row DataFrame."""
    feat_rows, mols, bit_infos, valid_rows = [], [], [], []
    for _, row in df_sub.iterrows():
        mol = Chem.MolFromSmiles(row["smi"])
        if mol is None:
            continue
        mol = Chem.AddHs(mol)

        feature_values = []
        mw = rdMolDescriptors.CalcExactMolWt(mol)
        feature_values.append(mw)
        feature_values.append(row["mass"] - mw)
        feature_values.append(row["z"])
        feature_values.append(rdMolDescriptors.CalcLabuteASA(mol))

        ohe = [0] * (len(ADDUCTS) + 1)
        ohe[ADDUCTS.index(row["adduct"]) if row["adduct"] in ADDUCTS else len(ADDUCTS)] = 1
        feature_values.extend(ohe)

        count_fp = morgan_count_fpgen.GetCountFingerprint(mol)
        arr = np.zeros((count_fp.GetLength(),), dtype=int)
        DataStructs.ConvertToNumpyArray(count_fp, arr)
        feature_values.extend(arr.tolist())

        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        morgan_bit_fpgen.GetFingerprint(mol, additionalOutput=ao)

        feat_rows.append(np.array(feature_values))
        mols.append(mol)
        bit_infos.append(ao.GetBitInfoMap())
        valid_rows.append(row)

    return feat_rows, mols, bit_infos, pd.DataFrame(valid_rows).reset_index(drop=True)

def compute_shap_batch(booster, X_batch, feature_names):
    dm = xgb.DMatrix(X_batch, feature_names=feature_names)
    shap_matrix = booster.predict(dm, pred_contribs=True)
    return shap_matrix[:, :-1], float(shap_matrix[0, -1])

def ensure_shap(start, end):
    """Compute SHAP only for indices in [start, end) not yet cached."""
    missing = [i for i in range(start, end) if i not in st.session_state.shap_cache]
    if not missing:
        return
    X_batch = st.session_state.X_full[missing]
    shap_vals, _ = compute_shap_batch(booster, X_batch, feature_names)
    for local_i, global_i in enumerate(missing):
        st.session_state.shap_cache[global_i] = shap_vals[local_i]

def draw_molecule(mol, active_bits, bit_info, size=(IMG_W, IMG_H)):
    highlight_atoms, highlight_bonds = {}, {}
    for bit_idx, bit in enumerate(active_bits):
        if bit not in bit_info:
            continue
        color = BIT_COLORS[bit_idx % len(BIT_COLORS)]
        for atom_idx, radius in bit_info[bit]:
            env  = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
            amap = {}
            Chem.PathToSubmol(mol, env, atomMap=amap)
            for a in amap.keys():
                if a not in highlight_atoms:
                    highlight_atoms[a] = color
            for b in env:
                if b not in highlight_bonds:
                    highlight_bonds[b] = color

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareMolForDrawing(mol)
    if highlight_atoms:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(highlight_atoms.keys()),
            highlightBonds=list(highlight_bonds.keys()),
            highlightAtomColors=highlight_atoms,
            highlightBondColors=highlight_bonds,
        )
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")

def make_waterfall(shap_row, global_base, subclass_base, feature_names,
                   true_ccs, pred_ccs, size=(WATERFALL_W, 420)):
    shift    = global_base - subclass_base
    adjusted = shap_row.copy()
    adjusted[0] += shift
    exp = shap.Explanation(
        values=adjusted,
        base_values=subclass_base,
        feature_names=feature_names,
    )
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100))
    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

# ── Session state init ─────────────────────────────────────────────────────────
defaults = {
    "computed_subclass": None,
    "X_full":            None,
    "mols":              None,
    "bit_infos":         None,
    "df_pool":           None,
    "global_base":       None,
    "subclass_base":     None,
    "shap_cache":        {},
    "pred_cache":        None,
    "n_shown":           PAGE_SIZE,
    "active_bits":       [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Static data ────────────────────────────────────────────────────────────────
model      = load_model()
booster    = model.get_booster()
df_counts, ADDUCTS = load_subclass_counts()

feature_names = (
    ["MolecularWeight", "AdductMass", "Charge", "LabuteASA"]
    + [f"Adduct_{sanitize(a)}" for a in ADDUCTS]
    + ["Adduct_other"]
    + [f"MorganFP_{i}" for i in range(1024)]
)

# ── Left sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Subclass")
    st.caption("Ranked by dataset size. Select one and click Apply.")
    subclass_labels = [
        f"{row['subclass']}  ({int(row['cnt'])})"
        for _, row in df_counts.iterrows()
    ]
    selected_label = st.radio("Subclass", subclass_labels, label_visibility="collapsed")
    apply_subclass = st.button("▶  Apply Subclass", use_container_width=True)

selected_subclass = selected_label.rsplit("  (", 1)[0]

# ── Layout ─────────────────────────────────────────────────────────────────────
body_col, right_col = st.columns([5, 1], gap="large")

with right_col:
    st.markdown("### 🧩 Fingerprint Bits")
    st.caption("Select up to 5 bits to highlight.")
    raw_selected = st.multiselect(
        "Bits", options=list(range(1024)),
        default=st.session_state.active_bits,
        format_func=lambda x: f"MorganFP_{x}",
        label_visibility="collapsed",
    )
    if len(raw_selected) > 5:
        st.warning("Max 5 — only first 5 used.")
    if st.button("▶  Apply Bits", use_container_width=True):
        st.session_state.active_bits = raw_selected[:5]

    if st.session_state.active_bits:
        st.markdown("**Legend:**")
        for i, bit in enumerate(st.session_state.active_bits):
            st.markdown(
                f'<span style="color:{BIT_COLOR_HEX[i]};font-size:18px">■</span> MorganFP_{bit}',
                unsafe_allow_html=True,
            )

# ── Apply Subclass ─────────────────────────────────────────────────────────────
if apply_subclass and selected_subclass != st.session_state.computed_subclass:
    with body_col:
        with st.spinner(f"Featurising all molecules in **{selected_subclass}**…"):
            df_sub = load_subclass_data(selected_subclass)
            feat_rows, mols, bit_infos, df_pool = featurise_all(df_sub, ADDUCTS)
            X_full = np.vstack(feat_rows)

        with st.spinner("Computing predictions for subclass baseline…"):
            # Fast regular predict over all mols to get subclass baseline
            dm_all   = xgb.DMatrix(X_full, feature_names=feature_names)
            preds    = booster.predict(dm_all)
            sub_base = float(preds.mean())

            # Get global base value from a single pred_contribs call
            dm_one      = xgb.DMatrix(X_full[:1], feature_names=feature_names)
            shap_one    = booster.predict(dm_one, pred_contribs=True)
            global_base = float(shap_one[0, -1])

        st.session_state.computed_subclass = selected_subclass
        st.session_state.X_full            = X_full
        st.session_state.mols              = mols
        st.session_state.bit_infos         = bit_infos
        st.session_state.df_pool           = df_pool
        st.session_state.global_base       = global_base
        st.session_state.subclass_base     = sub_base
        st.session_state.pred_cache        = preds
        st.session_state.shap_cache        = {}       # clear old SHAP cache
        st.session_state.n_shown           = PAGE_SIZE

# ── Body ───────────────────────────────────────────────────────────────────────
with body_col:
    st.markdown("# 🧪 CCS Fingerprint Explorer")

    if st.session_state.computed_subclass is None:
        st.info("👈  Select a subclass and click **Apply Subclass** to begin.")
    else:
        n        = len(st.session_state.mols)
        sub_base = st.session_state.subclass_base
        n_shown  = min(st.session_state.n_shown, n)

        st.markdown(f"### {st.session_state.computed_subclass}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total molecules",       n)
        c2.metric("Showing",               n_shown)
        c3.metric("Subclass baseline CCS", f"{sub_base:.2f} Å²")
        c4.metric("Bits highlighted",      len(st.session_state.active_bits))
        st.divider()

        # Compute SHAP on demand — only for the current visible window
        with st.spinner(f"Computing SHAP for molecules {n_shown - PAGE_SIZE + 1}–{n_shown}…"):
            ensure_shap(0, n_shown)

        active_bits = st.session_state.active_bits
        df_pool     = st.session_state.df_pool
        preds       = st.session_state.pred_cache

        for i in range(n_shown):
            mol      = st.session_state.mols[i]
            bit_info = st.session_state.bit_infos[i]
            shap_row = st.session_state.shap_cache[i]
            true_ccs = float(df_pool.loc[i, "ccs"])
            pred_ccs = float(preds[i])
            err      = pred_ccs - true_ccs
            smi      = df_pool.loc[i, "smi"]

            label = (f"Molecule {i+1}  |  True: {true_ccs:.2f}  |  "
                     f"Pred: {pred_ccs:.2f}  |  Err: {err:+.2f}")

            with st.expander(label, expanded=True):
                mol_col, wf_col = st.columns([1, 1], gap="medium")

                with mol_col:
                    mol_img = draw_molecule(mol, active_bits, bit_info, size=(IMG_W, IMG_H))
                    st.image(mol_img, use_container_width=True)
                    st.caption(smi[:80] + ("…" if len(smi) > 80 else ""))

                with wf_col:
                    wf_buf = make_waterfall(
                        shap_row,
                        st.session_state.global_base,
                        sub_base,
                        feature_names,
                        true_ccs, pred_ccs,
                        size=(WATERFALL_W, 420),
                    )
                    st.image(wf_buf, use_container_width=True)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Subclass baseline", f"{sub_base:.2f}")
                    m2.metric("True CCS",          f"{true_ccs:.2f}")
                    m3.metric("Predicted CCS",     f"{pred_ccs:.2f}",
                              delta=f"{err:+.2f}", delta_color="inverse")

        st.divider()

        if n_shown < n:
            remaining = n - n_shown
            if st.button(
                f"⬇  Load {min(PAGE_SIZE, remaining)} more  ({n_shown} / {n} shown)",
                use_container_width=True,
            ):
                st.session_state.n_shown += PAGE_SIZE
                st.rerun()
        else:
            st.success(f"✅  All {n} molecules shown.")