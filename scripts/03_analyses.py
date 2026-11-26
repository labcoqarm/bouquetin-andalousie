# -*- coding: utf-8 -*-
"""
Étape 3 : analyses statistiques (les figures sont faites par 04_figures.py).
H1 productivité (P), H2 stress (M), H3 stabilité (V) testées chacune avec sa composante, comme dans Radeloff et al. (2019),
puis autocorrélation, points chauds, routes, modèle de présence et corridors.
Sorties : resultats/*.csv, sorties/rasters/*.tif, sorties/echantillons.csv, couches ajoutées à sorties/couches.gpkg
"""

import os
import warnings
from itertools import combinations

import geopandas as gpd
import numpy as np
import pandas as pd
from esda.getisord import G_Local
from esda.moran import Moran
from libpysal.weights import DistanceBand, Queen
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from scipy.stats import kendalltau, mannwhitneyu, spearmanr
from shapely.geometry import LineString
from skimage.graph import route_through_array
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from commun import ANNEES, COUCHES, GRAINE, PIXEL, RASTERS, REFERENCE, RESULTATS, SORTIES, ecrire_tif, lire_tif, valeurs_sous_points

# Hors de la zone, les moyennes portent sur des pixels tous vides : avertissement attendu
warnings.filterwarnings("ignore", message="Mean of empty slice")
# Avertissement interne de scikit-learn sous Windows, sans effet sur les résultats
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
synthese = {}


def noter(cle, valeur):
    synthese[cle] = valeur
    print(f"  {cle} = {valeur:.4g}" if isinstance(valeur, float) else f"  {cle} = {valeur}")


def tableau(df, nom):
    df.to_csv(os.path.join(RESULTATS, nom), index=False, float_format="%.5g")


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
ref, profil = lire_tif(REFERENCE)
dedans = np.isfinite(ref)
xmin, ymax = profil["transform"].c, profil["transform"].f

zone = gpd.read_file(COUCHES, layer="zone")
provinces = gpd.read_file(COUCHES, layer="provinces")
obs = gpd.read_file(COUCHES, layer="observations")
obs_1km = gpd.read_file(COUCHES, layer="observations_1km")
fond = gpd.read_file(COUCHES, layer="fond")
chasse = gpd.read_file(COUCHES, layer="chasse")
g20 = gpd.read_file(COUCHES, layer="grille_20km")
g5 = gpd.read_file(COUCHES, layer="grille_5km")
print(f"{len(obs)} observations, {len(obs_1km)} mailles de 1 km occupées, {len(fond)} points de fond, {len(chasse)} mailles de chasse")

pile = {c: np.stack([lire_tif(os.path.join(RASTERS, f"{c}_{a}.tif"))[0] for a in ANNEES]) for c in ("P", "M", "V", "NM")}

# Pixel retenu une année donnée : dans la zone, au moins 9 mois sans nuages, pas en eau (NDVI moyen > 0)
valide = dedans[None] & (pile["NM"] >= 9) & (pile["P"] > 0)
for c in ("P", "M", "V"):
    pile[c] = np.where(valide, pile[c], np.nan)
noter("part_pixels_valides", float(valide.sum() / (dedans.sum() * len(ANNEES))))

# Normalisation sur toute la période (2e-98e centiles), pour que les années restent comparables
bornes = {c: np.nanpercentile(pile[c], [2, 98]) for c in ("P", "M", "V")}
norm = {c: np.clip((pile[c] - lo) / (hi - lo), 0, 1) for c, (lo, hi) in bornes.items()}
dhi = 0.5 * norm["P"] + 0.3 * norm["M"] + 0.2 * (1 - norm["V"])

moy = {c: np.nanmean(pile[c], axis=0) for c in ("P", "M", "V")}
dhi_moyen = np.nanmean(dhi, axis=0)
delta = dhi[ANNEES.index(2024)] - dhi[ANNEES.index(2017)]
altitude, _ = lire_tif(os.path.join(RASTERS, "altitude.tif"))
gy, gx = np.gradient(altitude, PIXEL)
pente = np.degrees(np.arctan(np.hypot(gx, gy)))
dist_routes, _ = lire_tif(os.path.join(RASTERS, "distance_routes.tif"))
for arr in (altitude, pente, dist_routes):
    arr[~dedans] = np.nan

for nom, arr in (("P_moyen", moy["P"]), ("M_moyen", moy["M"]), ("V_moyen", moy["V"]), ("DHI_moyen", dhi_moyen), ("DHI_delta_2024_2017", delta), ("pente", pente)):
    ecrire_tif(os.path.join(RASTERS, f"{nom}.tif"), arr, profil)

# ---------------------------------------------------------------------------
# 1. Évolution 2017-2024
# ---------------------------------------------------------------------------
print("1. Évolution du DHI")
annuel = pd.DataFrame({
    "annee": ANNEES,
    "P": np.nanmean(pile["P"], axis=(1, 2)),
    "M": np.nanmean(pile["M"], axis=(1, 2)),
    "V": np.nanmean(pile["V"], axis=(1, 2)),
    "DHI": np.nanmean(dhi, axis=(1, 2)),
    "part_valide": valide.sum(axis=(1, 2)) / dedans.sum(),
    "observations": obs["annee"].value_counts().reindex(ANNEES, fill_value=0).to_numpy(),
})
tableau(annuel, "dhi_annuel.csv")
tau, p_tau = kendalltau(annuel["annee"], annuel["DHI"])
noter("DHI_2017", float(annuel["DHI"].iloc[0]))
noter("DHI_2024", float(annuel["DHI"].iloc[-1]))
noter("tendance_kendall_tau", float(tau))
noter("tendance_kendall_p", float(p_tau))
d = delta[np.isfinite(delta)]
noter("part_baisse_delta_inf_-0.02", float((d < -0.02).mean()))
noter("part_stable_delta_0.02", float((np.abs(d) <= 0.02).mean()))
noter("part_hausse_delta_sup_0.02", float((d > 0.02).mean()))

# ---------------------------------------------------------------------------
# 2. Utilisation et disponibilité : chaque composante testée contre le fond
# ---------------------------------------------------------------------------
print("2. Présences contre points de fond")
variables = {
    "P_moyen": (moy["P"], "P, productivité cumulée (H1)", "NDVI cumulé"),
    "M_moyen": (moy["M"], "M, productivité minimale (H2)", "NDVI"),
    "V_moyen": (moy["V"], "V, variabilité (H3)", "coefficient de variation"),
    "DHI_moyen": (dhi_moyen, "DHI synthétique", "sans unité"),
    "altitude": (altitude, "Altitude", "m"),
    "pente": (pente, "Pente", "degrés"),
    "dist_routes": (dist_routes, "Distance aux routes principales", "m"),
}
lignes = []
echantillons = {}
for nom, (arr, libelle, unite) in variables.items():
    vp = valeurs_sous_points(obs_1km, arr, profil)
    vf = valeurs_sous_points(fond, arr, profil)
    vp, vf = vp[np.isfinite(vp)], vf[np.isfinite(vf)]
    u, p = mannwhitneyu(vp, vf)
    echantillons[nom] = (vp, vf)
    # AUC = probabilité qu'une présence ait une valeur plus forte qu'un point de fond (0,5 = aucune sélection)
    lignes.append({"variable": nom, "libelle": libelle, "unite": unite, "mediane_presences": np.median(vp), "mediane_fond": np.median(vf), "auc": u / (len(vp) * len(vf)), "p_mann_whitney": p, "n_presences": len(vp), "n_fond": len(vf)})
ua = pd.DataFrame(lignes)
tableau(ua, "utilisation_disponibilite.csv")
print(ua[["variable", "mediane_presences", "mediane_fond", "auc"]].to_string(index=False))

# Même test à 5 km avec les tableaux de chasse : mailles avec prélèvement contre toutes les mailles
g5_ua = gpd.read_file(COUCHES, layer="grille_5km")
g5_ua = g5_ua[g5_ua["part_zone"] >= 0.5].reset_index(drop=True)
ids5 = rasterize(((g, i) for i, g in enumerate(g5_ua.geometry)), out_shape=dedans.shape, transform=profil["transform"], fill=-1, dtype=np.int32)
chassee = np.zeros(len(g5_ua), dtype=bool)
jointure = gpd.sjoin(chasse[["geometry"]], g5_ua[["geometry"]], predicate="within")
chassee[jointure["index_right"].to_numpy()] = True
lignes = []
for nom, (arr, libelle, unite) in variables.items():
    ok = (ids5 >= 0) & np.isfinite(arr)
    nombre = np.bincount(ids5[ok], minlength=len(g5_ua))
    moyenne = np.where(nombre > 0, np.bincount(ids5[ok], weights=arr[ok], minlength=len(g5_ua)) / np.maximum(nombre, 1), np.nan)
    vp, vf = moyenne[chassee], moyenne
    vp, vf = vp[np.isfinite(vp)], vf[np.isfinite(vf)]
    u, p = mannwhitneyu(vp, vf)
    echantillons[f"{nom}_5km"] = (vp, vf)
    lignes.append({"variable": nom, "libelle": libelle, "unite": unite, "mediane_presences": np.median(vp), "mediane_fond": np.median(vf), "auc": u / (len(vp) * len(vf)), "p_mann_whitney": p, "n_presences": len(vp), "n_fond": len(vf)})
ua5 = pd.DataFrame(lignes)
tableau(ua5, "utilisation_disponibilite_chasse_5km.csv")
noter("mailles_5km_avec_chasse", int(chassee.sum()))
print(ua5[["variable", "mediane_presences", "mediane_fond", "auc"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Grille de 20 km et provinces : corrélations et I de Moran
# ---------------------------------------------------------------------------
print("3. Grille de 20 km et provinces")


def moyennes_zonales(couche, arr):
    """Moyenne des pixels valides de chaque entité (identifiant = index de la couche)."""
    ids = rasterize(((g, i) for i, g in enumerate(couche.geometry)), out_shape=arr.shape, transform=profil["transform"], fill=-1, dtype=np.int32)
    ok = (ids >= 0) & np.isfinite(arr)
    somme = np.bincount(ids[ok], weights=arr[ok], minlength=len(couche))
    nombre = np.bincount(ids[ok], minlength=len(couche))
    return np.where(nombre > 0, somme / np.maximum(nombre, 1), np.nan)


def compter(couche, points):
    jointure = gpd.sjoin(points[["geometry"]], couche[["geometry"]], predicate="within")
    return jointure.groupby("index_right").size().reindex(range(len(couche)), fill_value=0).to_numpy()


for c in ("P", "M", "V"):
    g20[f"{c}_moyen"] = moyennes_zonales(g20, moy[c])
g20["DHI_moyen"] = moyennes_zonales(g20, dhi_moyen)
g20["nb_obs"] = compter(g20, obs)
g20["nb_mailles_1km"] = compter(g20, obs_1km)
g20 = g20.dropna(subset=["DHI_moyen"]).reset_index(drop=True)
noter("cellules_20km", len(g20))
noter("cellules_20km_sans_observation", int((g20["nb_obs"] == 0).sum()))

lignes = []
for col in ("P_moyen", "M_moyen", "V_moyen", "DHI_moyen"):
    rho, p = spearmanr(g20[col], g20["nb_mailles_1km"])
    lignes.append({"variable": col, "spearman_rho": rho, "p": p, "n_cellules": len(g20)})
tableau(pd.DataFrame(lignes), "correlations_grille_20km.csv")
print(pd.DataFrame(lignes).to_string(index=False))

w = Queen.from_dataframe(g20, use_index=False, silence_warnings=True)
w.transform = "r"
for col, cle in ((np.log1p(g20["nb_obs"]), "observations"), (g20["DHI_moyen"], "DHI")):
    mi = Moran(col.to_numpy(), w, permutations=999)
    noter(f"moran_grille_{cle}_I", float(mi.I))
    noter(f"moran_grille_{cle}_p", float(mi.p_sim))

provinces["nb_obs"] = compter(provinces, obs)
provinces["DHI_moyen"] = moyennes_zonales(provinces, dhi_moyen)
provinces["part_obs"] = provinces["nb_obs"] / len(obs)
tableau(provinces.drop(columns="geometry").sort_values("nb_obs", ascending=False), "provinces.csv")
wp = Queen.from_dataframe(provinces, use_index=False, silence_warnings=True)
wp.transform = "r"
mp = Moran(provinces["nb_obs"].to_numpy(dtype=float), wp, permutations=999)
noter("moran_provinces_I", float(mp.I))
noter("moran_provinces_p", float(mp.p_sim))
g20.to_file(COUCHES, layer="grille_20km_resultats", driver="GPKG")

# ---------------------------------------------------------------------------
# 4. Points chauds Getis-Ord Gi* sur le nombre d'observations par maille de 5 km
# ---------------------------------------------------------------------------
print("4. Points chauds Gi*")
g5["nb_obs"] = compter(g5, obs)
centres = np.column_stack([g5.centroid.x, g5.centroid.y])
w5 = DistanceBand(centres, threshold=10_000, binary=True, silence_warnings=True)
gi = G_Local(g5["nb_obs"].to_numpy(dtype=float), w5, transform="B", permutations=999, star=True, seed=GRAINE)
g5["gi_z"] = gi.Zs
g5["gi_p"] = gi.p_sim
g5["point_chaud"] = (g5["gi_z"] > 1.96) & (g5["gi_p"] < 0.05)
g5.to_file(COUCHES, layer="points_chauds_5km", driver="GPKG")
noter("mailles_5km_points_chauds", int(g5["point_chaud"].sum()))
noter("part_obs_en_point_chaud", float(g5.loc[g5["point_chaud"], "nb_obs"].sum() / len(obs)))

# ---------------------------------------------------------------------------
# 5. Distance aux routes principales : observations contre territoire disponible
# ---------------------------------------------------------------------------
print("5. Routes principales")
d_obs = valeurs_sous_points(obs, dist_routes, profil)
d_1km = valeurs_sous_points(obs_1km, dist_routes, profil)
d_zone = dist_routes[dedans & np.isfinite(dist_routes)]
lignes = []
for seuil in (500, 2000, 5000, 10000):
    lignes.append({"seuil_m": seuil, "part_observations": np.nanmean(d_obs < seuil), "part_mailles_1km": np.nanmean(d_1km < seuil), "part_territoire": np.mean(d_zone < seuil)})
routes_df = pd.DataFrame(lignes)
tableau(routes_df, "routes.csv")
print(routes_df.to_string(index=False))
noter("distance_routes_mediane_obs_m", float(np.nanmedian(d_obs)))
noter("distance_routes_mediane_territoire_m", float(np.median(d_zone)))

# ---------------------------------------------------------------------------
# 6. Modèle de présence (forêt aléatoire) avec validation croisée spatiale
# ---------------------------------------------------------------------------
print("6. Forêt aléatoire")
predicteurs = ["P_moyen", "M_moyen", "V_moyen", "altitude", "pente", "dist_routes"]
rasters_pred = {nom: variables[nom][0] for nom in predicteurs}


def tableau_points(points, classe):
    t = pd.DataFrame({nom: valeurs_sous_points(points, arr, profil) for nom, arr in rasters_pred.items()})
    t["presence"] = classe
    t["bloc"] = (points.geometry.x // 50_000).astype(int).astype(str).to_numpy() + "_" + (points.geometry.y // 50_000).astype(int).astype(str).to_numpy()
    return t


donnees = pd.concat([tableau_points(obs_1km, 1), tableau_points(fond, 0)]).dropna().reset_index(drop=True)
X, y, blocs = donnees[predicteurs], donnees["presence"].to_numpy(), donnees["bloc"].to_numpy()
noter("rf_presences", int(y.sum()))
noter("rf_fond", int((y == 0).sum()))


def foret():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=GRAINE)


# Blocs de 50 km : le modèle est évalué sur des zones qu'il n'a pas vues
auc_spatial, importances = [], []
prob_oof = np.full(len(y), np.nan)
for app, test in GroupKFold(n_splits=5).split(X, y, blocs):
    rf = foret().fit(X.iloc[app], y[app])
    prob_oof[test] = rf.predict_proba(X.iloc[test])[:, 1]
    auc_spatial.append(roc_auc_score(y[test], prob_oof[test]))
    imp = permutation_importance(rf, X.iloc[test], y[test], scoring="roc_auc", n_repeats=5, random_state=GRAINE)
    importances.append(imp.importances_mean)

# Découpage aléatoire, pour mesurer l'optimisme d'une validation qui ignore l'espace
auc_aleatoire = []
for app, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=GRAINE).split(X, y):
    rf = foret().fit(X.iloc[app], y[app])
    auc_aleatoire.append(roc_auc_score(y[test], rf.predict_proba(X.iloc[test])[:, 1]))

noter("rf_auc_spatial_moyenne", float(np.mean(auc_spatial)))
noter("rf_auc_spatial_ecart_type", float(np.std(auc_spatial)))
noter("rf_auc_aleatoire_moyenne", float(np.mean(auc_aleatoire)))
imp_df = pd.DataFrame({"variable": predicteurs, "importance_moyenne": np.mean(importances, axis=0), "importance_ecart_type": np.std(importances, axis=0)}).sort_values("importance_moyenne", ascending=False)
tableau(imp_df, "random_forest_importance.csv")
print(imp_df.to_string(index=False))

# Seuil P10 : 90 % des présences, prédites hors de leur bloc, sont au-dessus
seuil = float(np.percentile(prob_oof[y == 1], 10))
noter("rf_seuil_p10", seuil)

rf = foret().fit(X, y)
pile_pred = np.stack([rasters_pred[n] for n in predicteurs], axis=-1)
complet = dedans & np.all(np.isfinite(pile_pred), axis=-1)
proba = np.full(dedans.shape, np.nan, dtype=np.float32)
lignes_ok = pile_pred[complet]
proba_ok = np.empty(len(lignes_ok), dtype=np.float32)
for debut in range(0, len(lignes_ok), 200_000):
    proba_ok[debut:debut + 200_000] = rf.predict_proba(pd.DataFrame(lignes_ok[debut:debut + 200_000], columns=predicteurs))[:, 1]
proba[complet] = proba_ok
ecrire_tif(os.path.join(RASTERS, "probabilite_presence.tif"), proba, profil)

# Habitat favorable à plus de 10 km de toute maille occupée : potentiel de recolonisation
occupe = np.zeros(dedans.shape, dtype=bool)
lig = ((ymax - obs_1km.geometry.y) // PIXEL).astype(int).clip(0, dedans.shape[0] - 1)
col = ((obs_1km.geometry.x - xmin) // PIXEL).astype(int).clip(0, dedans.shape[1] - 1)
occupe[lig, col] = True
dist_obs = distance_transform_edt(~occupe) * PIXEL
favorable = complet & (proba >= seuil)
recolonisable = favorable & (dist_obs > 10_000)
ecrire_tif(os.path.join(RASTERS, "potentiel_recolonisation.tif"), np.where(complet, recolonisable.astype(np.float32), np.nan), profil)
km2 = (PIXEL / 1000) ** 2
noter("surface_favorable_km2", float(favorable.sum() * km2))
noter("surface_favorable_non_occupee_km2", float(recolonisable.sum() * km2))
ids_prov = rasterize(((g, i) for i, g in enumerate(provinces.geometry)), out_shape=dedans.shape, transform=profil["transform"], fill=-1, dtype=np.int32)
provinces["recolonisable_km2"] = np.bincount(ids_prov[recolonisable & (ids_prov >= 0)], minlength=len(provinces)) * km2
tableau(provinces.drop(columns="geometry").sort_values("recolonisable_km2", ascending=False), "provinces.csv")

# ---------------------------------------------------------------------------
# 7. Corridors de moindre coût entre noyaux de population
# ---------------------------------------------------------------------------
print("7. Corridors")
# Résistance : habitat peu productif et voisinage des routes ; hors zone quasi infranchissable
resistance = 1 + 4 * (1 - dhi_moyen) + 5 * np.exp(-dist_routes / 500)
resistance = np.where(dedans & np.isnan(resistance), 50, resistance)
resistance = np.where(dedans, resistance, 10_000)
ecrire_tif(os.path.join(RASTERS, "resistance.tif"), np.where(dedans, resistance, np.nan), profil)

# Noyaux : groupes de mailles en point chaud, les six plus peuplés
chauds = g5[g5["point_chaud"]]
if chauds.empty:
    raise ValueError("Aucune maille en point chaud : impossible de définir des noyaux de population")
noyaux = chauds.dissolve().explode(index_parts=False).reset_index(drop=True)[["geometry"]]
noyaux["nb_obs"] = compter(noyaux, obs)
noyaux = noyaux[noyaux["nb_obs"] >= 50].sort_values("nb_obs", ascending=False).head(6).reset_index(drop=True)
noyaux["province"] = gpd.sjoin(gpd.GeoDataFrame(geometry=noyaux.representative_point(), crs=noyaux.crs), provinces[["province", "geometry"]], predicate="within").groupby(level=0)["province"].first().reindex(noyaux.index).to_numpy()
noter("noyaux_population", len(noyaux))
if len(noyaux) < 2:
    raise ValueError("Moins de deux noyaux de population : pas de corridor possible")
noyaux.to_file(COUCHES, layer="noyaux", driver="GPKG")

points_noyaux = []
for pt in noyaux.representative_point():
    points_noyaux.append((int((ymax - pt.y) // PIXEL), int((pt.x - xmin) // PIXEL)))

passages = np.zeros(dedans.shape, dtype=np.float32)
chemins = []
for (i, a), (j, b) in combinations(enumerate(points_noyaux), 2):
    pixels, cout = route_through_array(resistance, a, b, fully_connected=True, geometric=True)
    lig, col = np.array(pixels).T
    passages[lig, col] += 1
    ligne = LineString([(xmin + (c + 0.5) * PIXEL, ymax - (l + 0.5) * PIXEL) for l, c in pixels])
    chemins.append({"de": noyaux.loc[i, "province"], "vers": noyaux.loc[j, "province"], "noyau_de": i, "noyau_vers": j, "longueur_km": ligne.length / 1000, "cout": cout * PIXEL / 1000, "cout_par_km": cout * PIXEL / ligne.length, "geometry": ligne})
chemins = gpd.GeoDataFrame(chemins, crs=zone.crs)
chemins.to_file(COUCHES, layer="corridors", driver="GPKG")
ecrire_tif(os.path.join(RASTERS, "passages_corridors.tif"), np.where(dedans, passages, np.nan), profil)
tableau(chemins.drop(columns="geometry"), "corridors.csv")
print(chemins.drop(columns="geometry").to_string(index=False))

tableau(pd.DataFrame(list(synthese.items()), columns=["statistique", "valeur"]), "synthese.csv")

# Valeurs aux présences et sur le territoire, pour les figures
lignes = []
for cle, (vp, vf) in echantillons.items():
    nom, echelle = (cle[:-4], "5 km") if cle.endswith("_5km") else (cle, "1 km")
    lignes.append(pd.DataFrame({"variable": nom, "echelle": echelle, "groupe": "presence", "valeur": vp}))
    lignes.append(pd.DataFrame({"variable": nom, "echelle": echelle, "groupe": "territoire", "valeur": vf}))
pd.concat(lignes).to_csv(os.path.join(SORTIES, "echantillons.csv"), index=False, float_format="%.5g")
print(f"\nÉtape 3 terminée : {RESULTATS}. Figures : python 04_figures.py")

