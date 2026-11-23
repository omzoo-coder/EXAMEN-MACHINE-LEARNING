from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

st.set_page_config(page_title="Wholesale Customers", page_icon="🛒", layout="wide")

MODELES = Path(__file__).resolve().parent.parent / "models" / "wholesale_customers"


@st.cache_resource
def charger():
    """Charge une seule fois les fichiers joblib."""
    return {
        "centroides": joblib.load(MODELES / "wholesale_centroides.joblib"),
        "kmeans": joblib.load(MODELES / "wholesale_kmeans_model.joblib"),
        "pca": joblib.load(MODELES / "wholesale_pca_model.joblib"),
        "points": joblib.load(MODELES / "wholesale_points_pca.joblib"),
        "config": joblib.load(MODELES / "wholesale_config.joblib"),
    }


def predire(valeurs, m):
    """Prépare l'observation (même transformation qu'à l'entraînement), puis renvoie la classe du centroïde le plus proche."""
    obs = normalize(np.array(valeurs, dtype=float).reshape(1, -1))
    classe = m["config"]["classes"][cdist(obs, m["centroides"]).argmin()]
    return obs, int(classe)


m = charger()
cfg = m["config"]

# Sécurité : les fichiers du dossier models doivent être ceux de CE projet
ATTENDU = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
if list(cfg["colonnes"]) != ATTENDU:
    st.error("Les fichiers du dossier models/wholesale_customers ne correspondent pas à ce projet. "
             "Colonnes trouvées : " + ", ".join(map(str, cfg["colonnes"])) + ". "
             "Remplacez-les par ceux du notebook Wholesale Customers.")
    st.stop()

st.title("🛒 Wholesale Customers")
st.write("Saisissez les dépenses annuelles d'un client du grossiste pour connaître son groupe. Modèle déployé : **" + cfg["modele_final"] + "**.")

# ---------------------------------------------------------------- formulaire
with st.form("formulaire"):
    colonnes_ui = st.columns(3)
    valeurs = []
    for i, col in enumerate(cfg["colonnes"]):
        with colonnes_ui[i % 3]:
            valeurs.append(st.number_input(col, value=float(cfg["medianes"][col]),
                                           step=0.01, format="%.3f"))
    envoye = st.form_submit_button("Prédire la classe")

# ---------------------------------------------------------------- résultat
if envoye:
    obs, classe = predire(valeurs, m)
    st.success(f"Résultat : **{cfg['noms_classes'][classe]}**")

    # Position sur le plan de l'ACP
    pc = m["pca"].transform(obs)[0]
    fig = px.scatter(m["points"], x="PC1", y="PC2", color="Classe", opacity=0.6,
                     title="Position de l'observation sur le plan factoriel (ACP)")
    fig.add_scatter(x=[pc[0]], y=[pc[1]], mode="markers", name="Nouvelle observation",
                    marker=dict(size=16, color="black", symbol="star"))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- infos modèle
with st.expander("Profil moyen de chaque classe"):
    profil = cfg["profil"].copy()
    profil.index = [cfg["noms_classes"][i] for i in profil.index]
    st.dataframe(profil)

with st.expander("Qualité des modèles (indice de silhouette)"):
    st.write(pd.DataFrame(cfg["silhouette"], index=["Silhouette"]).T)
    st.write(f"Linkage de la CAH : {cfg.get('cah_linkage', '?')} | nombre de classes : {cfg.get('n_classes', '?')}")
    st.write(f"Variance expliquée par les 2 composantes de l'ACP : {cfg['variance_acp']} %")
    if "ari" in cfg:
        st.write(f"Accord avec les vraies classes du jeu de données (indice de Rand ajusté) : {cfg['ari']}")
