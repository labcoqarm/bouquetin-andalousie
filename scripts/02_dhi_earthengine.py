# -*- coding: utf-8 -*-
"""
Étape 2 : composantes du Dynamic Habitat Index (Radeloff et al., 2019) depuis Sentinel-2, et MNT, via Earth Engine.

Pour chaque année 2017-2024, à partir des 12 composites mensuels de NDVI :
- P, productivité cumulée : moyenne des mois disponibles x 12 (équivaut à la somme annuelle, sans pénaliser un mois manquant)
- M, productivité minimale : NDVI du mois le plus bas
- V, variabilité : coefficient de variation des NDVI mensuels
- NM, nombre de mois sans nuages, pour écarter les pixels mal couverts
Les composantes sont gardées brutes : la normalisation se fait ensuite sur toute la période (étape 3),
pour que les années restent comparables entre elles.

Prérequis : `earthengine authenticate` une fois, et la variable d'environnement EE_PROJECT (identifiant du projet Cloud).
Sorties : donnees/gee/*.tif (téléchargés), sorties/rasters/*.tif (recalés sur la grille de référence)
"""

import os

import ee
import geemap
import geopandas as gpd

from commun import ANNEES, COUCHES, CRS, GEE, PIXEL, RASTERS, REFERENCE, aligner, ecrire_tif, lire_tif

if "EE_PROJECT" not in os.environ:
    raise RuntimeError("Définir EE_PROJECT avec l'identifiant du projet Google Cloud associé à Earth Engine")
ee.Initialize(project=os.environ["EE_PROJECT"])

zone = gpd.read_file(COUCHES, layer="zone").to_crs("EPSG:4326")
aoi = ee.Geometry(zone.geometry.iloc[0].simplify(0.001).__geo_interface__)
emprise = ee.Geometry.Rectangle(list(zone.total_bounds))
_, profil_ref = lire_tif(REFERENCE)


def masquer_nuages(img):
    scl = img.select("SCL") # 3 ombre, 8-9 nuages, 10 cirrus
    return img.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))


def ndvi_mensuel(annee):
    mois = []
    for m in range(1, 13):
        debut = ee.Date.fromYMD(annee, m, 1)
        col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(debut, debut.advance(1, "month")).filterBounds(aoi).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).map(masquer_nuages)
        ndvi = col.map(lambda img: img.normalizedDifference(["B8", "B4"]))
        mois.append(ndvi.median().rename(f"m{m:02d}"))
    return ee.Image.cat(mois)


def composantes(annee):
    ndvi = ndvi_mensuel(annee)
    moyenne = ndvi.reduce(ee.Reducer.mean())
    ecart = ndvi.reduce(ee.Reducer.stdDev())
    return {
        "P": moyenne.multiply(12),
        "M": ndvi.reduce(ee.Reducer.min()),
        "V": ecart.divide(moyenne.max(0.01)),
        "NM": ndvi.reduce(ee.Reducer.count()),
    }


def exporter(image, nom):
    """Télécharge une bande à 250 m en UTM 30N, puis la recale sur la grille de référence."""
    brut = os.path.join(GEE, f"{nom}.tif")
    if not os.path.exists(brut):
        print(f"  export {nom}...")
        geemap.ee_export_image(image.clip(aoi).toFloat(), filename=brut, scale=PIXEL, crs=CRS, region=emprise, unmask_value=-9999)
    if not os.path.exists(brut):
        raise RuntimeError(f"Export Earth Engine échoué pour {nom}")
    ecrire_tif(os.path.join(RASTERS, f"{nom}.tif"), aligner(brut, profil_ref), profil_ref)


for annee in ANNEES:
    print(f"Composantes DHI {annee}")
    for nom, image in composantes(annee).items():
        exporter(image, f"{nom}_{annee}")

# Copernicus DEM GLO-30 à la place du GMRT du rapport : même source pour tout, et pente calculée en local
print("MNT")
dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(aoi).select("DEM").mosaic()
exporter(dem, "altitude")

print(f"\nÉtape 2 terminée : {RASTERS}")
