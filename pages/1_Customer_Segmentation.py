from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

st.set_page_config(page_title="Customer Segmentation", page_icon="📱", layout="wide")

MODELES = Path(__file__).resolve().parent.parent / "models" / "customer_segmentation"


@st.cache_resource
def charger():
    """Charge une seule fois les fichiers joblib."""
    return {
        "centroides": joblib.load(MODELES / "cah_centroides.joblib"),
        "kmeans": joblib.load(MODELES / "kmeans_model.joblib"),
        "pca": joblib.load(MODELES / "pca_model.joblib"),
        "points": joblib.load(MODELES / "points_pca.joblib"),
        "config": joblib.load(MODELES / "config.joblib"),
    }


def predire(valeurs, m):
    """Normalise le client, puis renvoie la classe du centroïde CAH le plus proche."""
    client = normalize(np.array(valeurs, dtype=float).reshape(1, -1))
    classe = m["config"]["classes"][cdist(client, m["centroides"]).argmin()]
    return client, classe


m = charger()
cfg = m["config"]

st.title("📱 Segmentation des clients : modèle CAH")
st.write("Saisissez les caractéristiques d'un client pour connaître sa classe.")

# ---------------------------------------------------------------- formulaire
with st.form("formulaire"):
    colonnes_ui = st.columns(3)
    valeurs = []
    for i, col in enumerate(cfg["colonnes"]):
        with colonnes_ui[i % 3]:
            valeurs.append(st.number_input(col, value=float(cfg["medianes"][col])))
    envoye = st.form_submit_button("Prédire la classe")

# ---------------------------------------------------------------- résultat
if envoye:
    client_norm, classe = predire(valeurs, m)
    nom = cfg["noms_classes"][classe]
    st.success(f"C'est un client **{nom}**")

    kmeans_classe = int(m["kmeans"].predict(client_norm)[0])
    st.caption(f"Pour comparaison, K-means (3 classes) le place dans la classe {kmeans_classe}.")

    # Position du client sur le plan de l'ACP
    pc = m["pca"].transform(client_norm)[0]
    fig = px.scatter(m["points"], x="PC1", y="PC2", color="Classe", opacity=0.6,
                     title="Position du client sur le plan factoriel (ACP)")
    fig.add_scatter(x=[pc[0]], y=[pc[1]], mode="markers", name="Nouveau client",
                    marker=dict(size=16, color="black", symbol="star"))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- infos modèle
with st.expander("Profil moyen de chaque classe"):
    profil = cfg["profil"].copy()
    profil.index = [cfg["noms_classes"][i] for i in profil.index]
    st.dataframe(profil)

with st.expander("Qualité des modèles (indice de silhouette)"):
    st.write(pd.DataFrame(cfg["silhouette"], index=["Silhouette"]).T)
    st.write(f"Variance expliquée par les 2 composantes de l'ACP : {cfg['variance_acp']} %")
