
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/readmission_model.joblib"

st.set_page_config(page_title="Risque de réadmission (30 jours)", layout="wide")

st.title("🏥 Risque de réadmission < 30 jours (démo)")
st.caption(
    "Outil de démonstration : aide à la priorisation du suivi post-sortie. "
    "Pas un outil de diagnostic médical."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Charger le modèle
try:
    model = load_model()
except Exception as e:
    st.error("Impossible de charger le modèle. As-tu bien lancé `python src/train.py` ?")
    st.exception(e)
    st.stop()

uploaded = st.file_uploader(
    "Charge un fichier CSV (mêmes colonnes que le dataset d’entraînement)",
    type=["csv"]
)

if not uploaded:
    st.info("Charge un CSV pour obtenir des scores. Tu peux tester avec data/diabetic_data.csv.")
    st.stop()

# Lire le fichier
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error("Impossible de lire le CSV.")
    st.exception(e)
    st.stop()

# Préparer X (on enlève la colonne cible si elle existe)
X = df.drop(columns=["readmitted"], errors="ignore")

# Prédire
try:
    proba = model.predict_proba(X)[:, 1]
except Exception as e:
    st.error(
        "Le modèle n'arrive pas à faire les prédictions. "
        "Vérifie que ton CSV a les mêmes colonnes que celui d'entraînement."
    )
    st.exception(e)
    st.stop()

out = df.copy()
out["risk_readmit_30"] = proba

# Affichage en 2 colonnes
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Aperçu des scores (premières lignes)")
    st.dataframe(out.head(50), use_container_width=True)

with col2:
    st.subheader("Indicateurs")
    st.metric("Nombre de patients", len(out))
    st.metric("Risque moyen", f"{out['risk_readmit_30'].mean():.2f}")
    seuil = st.slider("Seuil de priorisation", 0.0, 1.0, 0.50, 0.01)
    st.metric("Patients au-dessus du seuil", int((out["risk_readmit_30"] >= seuil).sum()))

st.subheader("Distribution des risques")
st.line_chart(out["risk_readmit_30"].sort_values().reset_index(drop=True))

st.divider()

# Encadré explication (simple et safe)
st.subheader("🧠 Pourquoi ce patient est à risque ? (explication indicative)")

st.caption(
    "Cette explication utilise des règles simples sur quelques variables du dataset "
    "pour rendre le score plus compréhensible. "
    "Ce n’est pas une interprétation médicale."
)

patient_index = st.number_input(
    "Choisis l’index du patient (ligne du fichier)",
    min_value=0,
    max_value=len(out) - 1,
    value=0,
    step=1
)

patient = out.iloc[int(patient_index)]

reasons = []

# Règles simples (POC). Elles marchent même si certaines colonnes n'existent pas.
age = patient.get("age", None)
if isinstance(age, str) and age not in ["[0-10]", "[10-20]", "[20-30]"]:
    reasons.append("Âge plutôt élevé (selon la tranche d'âge)")

number_inpatient = patient.get("number_inpatient", None)
if pd.notna(number_inpatient):
    try:
        if float(number_inpatient) > 1:
            reasons.append("Plusieurs hospitalisations précédentes")
    except Exception:
        pass

time_in_hospital = patient.get("time_in_hospital", None)
if pd.notna(time_in_hospital):
    try:
        if float(time_in_hospital) >= 7:
            reasons.append("Séjour hospitalier long")
    except Exception:
        pass

admission_type_id = patient.get("admission_type_id", None)
if pd.notna(admission_type_id):
    try:
        if int(float(admission_type_id)) in [1, 2]:
            reasons.append("Admission potentiellement non programmée / urgence")
    except Exception:
        pass

if len(reasons) == 0:
    st.info(
        "Aucun facteur simple ne ressort fortement. "
        "Le score est basé sur une combinaison statistique de plusieurs variables."
    )
else:
    st.write("Principaux signaux (règles simples) :")
    for r in reasons:
        st.write("•", r)

# Afficher le score du patient choisi
st.markdown(f"**Score de risque (risk_readmit_30) : {float(patient['risk_readmit_30']):.2f}**")
