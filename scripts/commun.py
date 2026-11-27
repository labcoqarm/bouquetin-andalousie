# -*- coding: utf-8 -*-
"""Chemins, constantes et fonctions raster partagés par les trois scripts."""

import os

# Les variables PROJ/GDAL laissées par QGIS cassent rasterio sous Windows ; celles de conda sont gardées
for var in ("PROJ_LIB", "GDAL_DATA", "GEOTIFF_CSV"):
    if "qgis" in os.environ.get(var, "").lower():
        os.environ.pop(var)

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRUTES = os.path.join(RACINE, "donnees", "brutes")
GEE = os.path.join(RACINE, "donnees", "gee")
SORTIES = os.path.join(RACINE, "sorties")
RASTERS = os.path.join(SORTIES, "rasters")
RESULTATS = os.path.join(RACINE, "resultats")
FIGURES = os.path.join(RACINE, "figures")
for dossier in (GEE, RASTERS, RESULTATS, FIGURES):
    os.makedirs(dossier, exist_ok=True)

GADM1 = os.path.join(BRUTES, "gadm", "gadm41_ESP_1.shp")
GADM2 = os.path.join(BRUTES, "gadm", "gadm41_ESP_2.shp")
ROUTES = os.path.join(BRUTES, "osm", "gis_osm_roads_free_1.shp")
DOSSIER_GBIF = os.path.join(BRUTES, "gbif")

COUCHES = os.path.join(SORTIES, "couches.gpkg")
REFERENCE = os.path.join(SORTIES, "reference_250m.tif")

CRS = "EPSG:25830" # ETRS89 / UTM 30N, distances en mètres
PIXEL = 250
ANNEES = list(range(2017, 2025))
NODATA = -9999.0
GRAINE = 2026


def lire_tif(chemin):
    """Première bande en float32, NODATA remplacé par NaN."""
    with rasterio.open(chemin) as src:
        return src.read(1, masked=True).astype(np.float32).filled(np.nan), src.profile.copy()


def ecrire_tif(chemin, arr, profil):
    profil = {**profil, "driver": "GTiff", "count": 1, "dtype": "float32", "nodata": NODATA, "compress": "deflate"}
    with rasterio.open(chemin, "w", **profil) as dst:
        dst.write(np.nan_to_num(arr, nan=NODATA).astype(np.float32), 1)


def aligner(source, profil_ref, methode=Resampling.bilinear):
    """Reprojette un raster sur la grille de référence (250 m, UTM 30N)."""
    sortie = np.full((profil_ref["height"], profil_ref["width"]), np.nan, dtype=np.float32)
    with rasterio.open(source) as src:
        reproject(source=rasterio.band(src, 1), destination=sortie, src_nodata=NODATA if src.nodata is None else src.nodata, dst_nodata=np.nan, dst_transform=profil_ref["transform"], dst_crs=profil_ref["crs"], resampling=methode)
    return sortie


def valeurs_sous_points(gdf, arr, profil):
    """Valeur du pixel sous chaque point (NaN hors grille)."""
    lig, col = rasterio.transform.rowcol(profil["transform"], gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy())
    lig, col = np.asarray(lig), np.asarray(col)
    dedans = (lig >= 0) & (lig < arr.shape[0]) & (col >= 0) & (col < arr.shape[1])
    vals = np.full(len(gdf), np.nan, dtype=np.float32)
    vals[dedans] = arr[lig[dedans], col[dedans]]
    return vals
