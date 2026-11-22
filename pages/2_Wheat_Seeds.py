from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.spatial.distance import cdist

st.set_page_config(page_title="Wheat Seeds", page_icon="🌾", layout="wide")

MODELES = Path(__file__).resolve().parent.parent / "models" / "wheat_seeds"


@st.cache_resource
def charger():
    """Charge une seule fois les fichiers joblib."""
    return {
        "centroides": joblib.load(MODELES / "centroides.joblib"),
        "kmeans": joblib.load(MODELES / "kmeans_model.joblib"),
        "pca": joblib.load(MODELES / "pca_model.joblib"),
        "points": joblib.load(MODELES / "points_pca.joblib"),
        "config": joblib.load(MODELES / "config.joblib"),
    }


def predire(valeurs, m):
    """Renvoie la graine (1 ligne) et la classe du centroïde le plus proche."""
    graine = np.array(valeurs, dtype=float).reshape(1, -1)
    classe = m["config"]["classes"][cdist(graine, m["centroides"]).argmin()]
    return graine, int(classe)


m = charger()
cfg = m["config"]

st.title("🌾 Classification des graines de blé")
st.write(f"Modèle déployé : **{cfg['modele_final']}**. "
         "Saisissez les 7 mesures d'une graine pour connaître son groupe.")

# ---------------------------------------------------------------- formulaire
with st.form("formulaire"):
    colonnes_ui = st.columns(3)
    valeurs = []
    for i, col in enumerate(cfg["colonnes"]):
        with colonnes_ui[i % 3]:
            valeurs.append(st.number_input(col, value=float(cfg["medianes"][col]),
                                           step=0.01, format="%.4f"))
    envoye = st.form_submit_button("Prédire le groupe")

# ---------------------------------------------------------------- résultat
if envoye:
    graine, classe = predire(valeurs, m)
    st.success(f"Cette graine appartient au groupe : **{cfg['noms_classes'][classe]}**")

    # Position de la graine sur le plan de l'ACP
    pc = m["pca"].transform(graine)[0]
    fig = px.scatter(m["points"], x="PCA1", y="PCA2", color="Classe", opacity=0.7,
                     title="Position de la graine sur le plan factoriel (ACP)")
    fig.add_scatter(x=[pc[0]], y=[pc[1]], mode="markers", name="Nouvelle graine",
                    marker=dict(size=16, color="black", symbol="star"))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- infos modèle
with st.expander("Profil moyen de chaque groupe"):
    profil = cfg["profil"].copy()
    profil.index = [cfg["noms_classes"][i] for i in profil.index]
    st.dataframe(profil)

with st.expander("Qualité des modèles (indice de silhouette)"):
    st.write(pd.DataFrame(cfg["silhouette"], index=["Silhouette"]).T)
    st.write(f"Linkage de la CAH : {cfg['cah_linkage']}")
    st.write(f"Variance expliquée par les 2 composantes de l'ACP : {cfg['variance_acp']} %")
