# -*- coding: utf-8 -*-
"""
Étape 1 : zone d'étude, observations GBIF, routes principales et grilles d'analyse.
Tout est reprojeté en ETRS89 / UTM 30N pour que distances et surfaces soient en mètres.
Sorties : sorties/couches.gpkg, sorties/reference_250m.tif, sorties/rasters/distance_routes.tif
"""

import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box

from commun import ANNEES, COUCHES, CRS, DOSSIER_GBIF, GADM1, GADM2, GRAINE, PIXEL, RASTERS, REFERENCE, ROUTES, ecrire_tif

for chemin in (GADM1, GADM2, ROUTES):
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Donnée absente : {chemin} (voir le README, section Données)")

# ---------------------------------------------------------------------------
# Zone d'étude et provinces (GADM 4.1)
# ---------------------------------------------------------------------------
regions = gpd.read_file(GADM1)
zone = regions[regions["NAME_1"].str.contains("Andaluc", na=False)].to_crs(CRS)
if len(zone) != 1:
    raise ValueError(f"{len(zone)} entités « Andalucía » trouvées dans {GADM1}")
zone = zone[["NAME_1", "geometry"]]
contour = zone.geometry.iloc[0]

provinces = gpd.read_file(GADM2)
provinces = provinces[provinces["NAME_1"].str.contains("Andaluc", na=False)].to_crs(CRS)[["NAME_2", "geometry"]]
provinces = provinces.rename(columns={"NAME_2": "province"}).reset_index(drop=True)
print(f"{len(provinces)} provinces")

zone.to_file(COUCHES, layer="zone", driver="GPKG")
provinces.to_file(COUCHES, layer="provinces", driver="GPKG")

# Grille de référence à 250 m calée sur des multiples de 250 m
xmin, ymin, xmax, ymax = contour.bounds
xmin, ymin = np.floor(xmin / PIXEL) * PIXEL, np.floor(ymin / PIXEL) * PIXEL
xmax, ymax = np.ceil(xmax / PIXEL) * PIXEL, np.ceil(ymax / PIXEL) * PIXEL
largeur, hauteur = int((xmax - xmin) / PIXEL), int((ymax - ymin) / PIXEL)
transform = from_origin(xmin, ymax, PIXEL, PIXEL)
profil = {"height": hauteur, "width": largeur, "crs": CRS, "transform": transform}
dedans = rasterize([(contour, 1)], out_shape=(hauteur, largeur), transform=transform, fill=0, dtype=np.uint8)
ecrire_tif(REFERENCE, np.where(dedans == 1, 1.0, np.nan), profil)
print(f"Grille de référence : {largeur} x {hauteur} pixels de {PIXEL} m")

# ---------------------------------------------------------------------------
# Observations GBIF 2017-2024
# ---------------------------------------------------------------------------
fichiers = glob.glob(os.path.join(DOSSIER_GBIF, "*.csv"))
if len(fichiers) != 1:
    raise FileNotFoundError(f"Il faut exactement un CSV GBIF dans {DOSSIER_GBIF}, trouvé : {fichiers}")
df = pd.read_csv(fichiers[0], sep="\t", low_memory=False, on_bad_lines="warn")
print(f"{len(df)} lignes dans le téléchargement GBIF")

df = df.dropna(subset=["decimalLatitude", "decimalLongitude", "year"])
df = df[df["year"].between(ANNEES[0], ANNEES[-1])]
print(f"  {len(df)} avec coordonnées et année {ANNEES[0]}-{ANNEES[-1]}")

# Tableaux de chasse (IREC-CSIC) : une ligne par animal ou lot déclaré, placée au centre d'une maille de 5 km
# de la grille européenne EPSG:3035, datée à l'année. Ce ne sont pas des observations ponctuelles : on les garde à part.
est_chasse = df["collectionCode"] == "BigGameBags"
chasse = df[est_chasse]
df = df[~est_chasse]
print(f"  {len(chasse)} lignes de tableaux de chasse mises à part, {len(df)} autres")

if "occurrenceStatus" in df.columns:
    df = df[df["occurrenceStatus"].fillna("PRESENT").str.upper() == "PRESENT"]
df = df[~df["basisOfRecord"].isin(["FOSSIL_SPECIMEN", "PRESERVED_SPECIMEN"])]
incertitude = pd.to_numeric(df["coordinateUncertaintyInMeters"], errors="coerce")
df = df[incertitude.isna() | (incertitude <= 1000)]
print(f"  {len(df)} présences vivantes, incertitude de position ≤ 1 km ou non renseignée")

# Un même point saisi plusieurs fois le même jour ne compte qu'une fois
df = df.drop_duplicates(subset=["decimalLatitude", "decimalLongitude", "eventDate"])
print(f"  {len(df)} après suppression des doublons (même point, même date)")

points = gpd.GeoDataFrame({"annee": df["year"].astype(int).to_numpy(), "gbifID": df["gbifID"].to_numpy()}, geometry=gpd.points_from_xy(df["decimalLongitude"], df["decimalLatitude"]), crs="EPSG:4326").to_crs(CRS)
obs = points[points.within(contour)].reset_index(drop=True)
print(f"  {len(obs)} en Andalousie")

# Une seule présence par maille de 1 km pour limiter le poids des sites très fréquentés
maille = (obs.geometry.x // 1000).astype(int).astype(str) + "_" + (obs.geometry.y // 1000).astype(int).astype(str)
obs_1km = obs.loc[~maille.duplicated()].reset_index(drop=True)
print(f"  {len(obs_1km)} mailles de 1 km occupées")

obs.to_file(COUCHES, layer="observations", driver="GPKG")
obs_1km.to_file(COUCHES, layer="observations_1km", driver="GPKG")

# Mailles de chasse : centre de maille en EPSG:3035, nombre d'années avec un prélèvement déclaré
centres = gpd.GeoDataFrame({"annee": chasse["year"].astype(int).to_numpy()}, geometry=gpd.points_from_xy(chasse["decimalLongitude"], chasse["decimalLatitude"]), crs="EPSG:4326").to_crs("EPSG:3035")
centres["x"], centres["y"] = centres.geometry.x.round(), centres.geometry.y.round()
mailles = centres.groupby(["x", "y"]).agg(nb_lignes=("annee", "size"), nb_annees=("annee", "nunique"), premiere=("annee", "min"), derniere=("annee", "max")).reset_index()
mailles = gpd.GeoDataFrame(mailles, geometry=gpd.points_from_xy(mailles["x"], mailles["y"]), crs="EPSG:3035").to_crs(CRS)
mailles = mailles[mailles.within(contour)].drop(columns=["x", "y"]).reset_index(drop=True)
mailles.to_file(COUCHES, layer="chasse", driver="GPKG")
print(f"  tableaux de chasse : {int(mailles['nb_lignes'].sum())} lignes dans {len(mailles)} mailles de 5 km en Andalousie")

# Points de fond : ce qui est disponible pour l'espèce dans la région
rng = np.random.default_rng(GRAINE)
lig, col = np.nonzero(dedans)
tirage = rng.choice(len(lig), size=10_000, replace=False)
x = xmin + (col[tirage] + rng.random(10_000)) * PIXEL
y = ymax - (lig[tirage] + rng.random(10_000)) * PIXEL
fond = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=CRS)
fond = fond[fond.within(contour)].reset_index(drop=True)
fond.to_file(COUCHES, layer="fond", driver="GPKG")
print(f"{len(fond)} points de fond")

# ---------------------------------------------------------------------------
# Routes principales (OSM) et distance en mètres
# ---------------------------------------------------------------------------
routes = gpd.read_file(ROUTES, columns=["fclass"])
routes = routes[routes["fclass"].isin(["motorway", "trunk", "primary"])].to_crs(CRS)
routes = routes[routes.intersects(contour.buffer(20_000))].reset_index(drop=True)
routes.to_file(COUCHES, layer="routes", driver="GPKG")
print(f"{len(routes)} tronçons de routes principales")

trace = rasterize(((g, 1) for g in routes.geometry), out_shape=(hauteur, largeur), transform=transform, fill=0, dtype=np.uint8, all_touched=True)
distance = distance_transform_edt(trace == 0) * PIXEL
ecrire_tif(os.path.join(RASTERS, "distance_routes.tif"), np.where(dedans == 1, distance, np.nan), profil)

# ---------------------------------------------------------------------------
# Grilles d'analyse
# ---------------------------------------------------------------------------
def grille(pas, crs):
    """Grille carrée calée sur des multiples du pas dans la projection donnée, ramenée en UTM 30N."""
    zone_crs = zone.to_crs(crs).geometry.iloc[0]
    x0, y0, x1, y1 = zone_crs.bounds
    x0, y0 = np.floor(x0 / pas) * pas, np.floor(y0 / pas) * pas
    cellules = [box(x, y, x + pas, y + pas) for x in np.arange(x0, x1, pas) for y in np.arange(y0, y1, pas)]
    g = gpd.GeoDataFrame(geometry=cellules, crs=crs)
    g = g[g.intersects(zone_crs)].to_crs(CRS).reset_index(drop=True)
    g["id"] = g.index
    g["part_zone"] = g.intersection(contour).area / g.area
    return g


# 20 km en UTM pour les statistiques ; 5 km sur la grille européenne EPSG:3035, celle des tableaux de chasse
for pas, crs, nom in ((20_000, CRS, "grille_20km"), (5_000, "EPSG:3035", "grille_5km")):
    g = grille(pas, crs)
    g.to_file(COUCHES, layer=nom, driver="GPKG")
    print(f"{nom} : {len(g)} cellules")

print(f"\nÉtape 1 terminée : {COUCHES}")
