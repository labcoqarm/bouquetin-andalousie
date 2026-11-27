# -*- coding: utf-8 -*-
"""
Étape 4 : cartes et graphiques, à partir des sorties de l'étape 3.
Sorties : figures/*.png
"""

import os

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

from commun import COUCHES, FIGURES, GADM1, PIXEL, RASTERS, REFERENCE, RESULTATS, SORTIES, lire_tif

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.edgecolor": "#888888",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.dpi": 200,
})

ROUGE = "#b2182b"
ORANGE = "#e08214"
GRIS = "#8c8c8c"
TEXTE = "#333333"
SOURCE = "Sources : Copernicus Sentinel-2 (Earth Engine), Copernicus DEM, GBIF, GADM 4.1, OpenStreetMap. ETRS89 / UTM 30N."

# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
ref, profil = lire_tif(REFERENCE)
dedans = np.isfinite(ref)
x0, y1 = profil["transform"].c, profil["transform"].f
etendue_raster = [x0, x0 + profil["width"] * PIXEL, y1 - profil["height"] * PIXEL, y1]


def raster(nom):
    arr, _ = lire_tif(os.path.join(RASTERS, f"{nom}.tif"))
    return np.where(dedans, arr, np.nan)


def couche(nom):
    return gpd.read_file(COUCHES, layer=nom)


zone = couche("zone")
provinces = couche("provinces")
obs = couche("observations")
chasse = couche("chasse")
g20 = couche("grille_20km_resultats")
g5 = couche("points_chauds_5km")
noyaux = couche("noyaux")
corridors = couche("corridors")
synthese = pd.read_csv(os.path.join(RESULTATS, "synthese.csv")).set_index("statistique")["valeur"]

# Cadre des cartes : le continent, sans l'île d'Alborán qui étirerait la carte vers le sud
parties = zone.explode(index_parts=False).reset_index(drop=True)
continent = parties.loc[[parties.area.idxmax()]]
trace_zone = parties[parties.area > 1e6]
# Limites de provinces sans les îlots (Alborán), qui faisaient un point isolé en bas des cartes
limites_provinces = provinces.explode(index_parts=False).reset_index(drop=True)
limites_provinces = limites_provinces[limites_provinces.area > 1e6].boundary
cx0, cy0, cx1, cy1 = continent.total_bounds
# Les pixels d'Alborán sortent aussi des rasters
y_pixels = y1 - (np.arange(profil["height"]) + 0.5) * PIXEL
dedans[y_pixels < cy0 - 5000, :] = False
marge = 12_000
cadre = (cx0 - marge, cx1 + marge, cy0 - marge, cy1 + marge)

altitude = raster("altitude")
ombrage = LightSource(azdeg=315, altdeg=40).hillshade(np.nan_to_num(altitude, nan=0), vert_exag=3, dx=PIXEL, dy=PIXEL)
ombrage = np.where(dedans, ombrage, np.nan)

LIBELLES = {
    "P_moyen": "P, productivité cumulée",
    "M_moyen": "M, productivité minimale",
    "V_moyen": "V, variabilité saisonnière",
    "DHI_moyen": "DHI synthétique",
    "altitude": "Altitude",
    "pente": "Pente",
    "dist_routes": "Distance aux routes",
}
UNITES = {"P_moyen": "NDVI cumulé sur l'année", "M_moyen": "NDVI minimal", "V_moyen": "coefficient de variation", "DHI_moyen": "sans unité", "altitude": "m", "pente": "degrés", "dist_routes": "km"}


# ---------------------------------------------------------------------------
# Habillage commun des cartes
# ---------------------------------------------------------------------------
def nouvelle_carte(largeur=10):
    hauteur = largeur * (cadre[3] - cadre[2]) / (cadre[1] - cadre[0])
    fig, ax = plt.subplots(figsize=(largeur, hauteur + 0.9))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.06, top=0.9)
    return fig, ax


def fond_relief(ax, alpha=1.0):
    ax.imshow(ombrage, cmap="Greys_r", extent=etendue_raster, vmin=0, vmax=1, alpha=alpha, interpolation="bilinear", zorder=0)


def habiller(ax, titre, sous_titre=None, limites=True):
    if limites:
        limites_provinces.plot(ax=ax, color="white", linewidth=0.9, zorder=8)
        limites_provinces.plot(ax=ax, color="#555555", linewidth=0.35, zorder=9)
    trace_zone.boundary.plot(ax=ax, color="#222222", linewidth=0.8, zorder=10)
    ax.set_xlim(cadre[0], cadre[1])
    ax.set_ylim(cadre[2], cadre[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    echelle(ax)
    nord(ax)
    ax.set_title(titre, loc="left", pad=18 if sous_titre else 6)
    if sous_titre:
        ax.text(0, 1.01, sous_titre, transform=ax.transAxes, fontsize=8.5, color="#555555", va="bottom")


def echelle(ax):
    """Barre d'échelle 0-25-50 km dans l'Atlantique, en bas à gauche."""
    x, y = cadre[0] + 18_000, cadre[2] + 16_000
    for i, couleur in enumerate(("#222222", "white")):
        ax.add_patch(mpatches.Rectangle((x + i * 25_000, y), 25_000, 3_000, facecolor=couleur, edgecolor="#222222", linewidth=0.6, zorder=20))
    for d in (0, 25, 50):
        ax.text(x + d * 1000, y - 2_500, f"{d}" if d < 50 else "50 km", ha="center", va="top", fontsize=7.5, color=TEXTE, zorder=20)


def nord(ax):
    x, y = cadre[0] + 22_000, cadre[3] - 20_000
    ax.annotate("", xy=(x, y), xytext=(x, y - 28_000), arrowprops={"arrowstyle": "-|>,head_width=0.35,head_length=0.7", "color": "#222222", "linewidth": 1.2}, zorder=20)
    ax.text(x, y + 3_000, "N", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222222", zorder=20)


def fond_sans_donnee(ax):
    """Gris sous les rasters : les pixels sans donnée (nuages persistants, neige, eau) restent visibles comme tels."""
    trace_zone.plot(ax=ax, color="#c8c8c8", edgecolor="none", zorder=0)


def barre_couleur(fig, ax, image, titre, ticks=None, format_=None, sans_donnee=True):
    """Barre de couleur horizontale posée dans la mer, en bas à droite de la carte."""
    cax = ax.inset_axes([0.62, 0.07, 0.3, 0.035])
    if sans_donnee:
        nd = ax.inset_axes([0.62, 0.0, 0.012, 0.025])
        nd.set_facecolor("#c8c8c8")
        nd.set_xticks([])
        nd.set_yticks([])
        for bord in nd.spines.values():
            bord.set_visible(False)
        ax.text(0.637, 0.0125, "sans donnée", transform=ax.transAxes, fontsize=7, color=TEXTE, va="center")
    cb = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=ticks, format=format_)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7.5, length=2)
    cax.set_title(titre, fontsize=8, color=TEXTE, loc="left", pad=3)
    return cb


def legende(ax, elements, titre=None, loc="lower right", ncol=1):
    leg = ax.legend(handles=elements, title=titre, loc=loc, ncol=ncol, fontsize=8, title_fontsize=8.5, bbox_to_anchor=(0.98, 0.04) if loc == "lower right" else None, borderaxespad=0.2, handlelength=1.6)
    leg._legend_box.align = "left"
    return leg


def enregistrer(fig, nom, source=SOURCE):
    fig.text(0.01, 0.012, source, fontsize=6.5, color="#666666")
    fig.savefig(os.path.join(FIGURES, nom), facecolor="white")
    plt.close(fig)
    print(f"  {nom}")


def etiquettes_provinces(ax, couleur=TEXTE):
    for _, p in provinces.iterrows():
        pt = p.geometry.representative_point()
        ax.text(pt.x, pt.y, p["province"].upper(), fontsize=6.5, ha="center", va="center", color=couleur, alpha=0.85, zorder=12, fontweight="bold", path_effects=[pe.withStroke(linewidth=2, foreground="white", alpha=0.7)])


print("Figures")

# ---------------------------------------------------------------------------
# 1. Situation
# ---------------------------------------------------------------------------
fig, ax = nouvelle_carte()
ax.imshow(np.where(dedans, altitude, np.nan), cmap="gist_earth", extent=etendue_raster, vmin=-300, vmax=3200, alpha=0.55, interpolation="bilinear", zorder=0)
ax.imshow(ombrage, cmap="Greys_r", extent=etendue_raster, alpha=0.35, interpolation="bilinear", zorder=1)
carres = chasse.copy()
carres["geometry"] = chasse.to_crs("EPSG:3035").buffer(2_500, cap_style=3).to_crs(zone.crs)
carres.plot(ax=ax, facecolor=ORANGE, edgecolor="none", alpha=0.45, zorder=3)
obs.plot(ax=ax, color=ROUGE, markersize=4, edgecolor="white", linewidth=0.2, zorder=5)
etiquettes_provinces(ax)
habiller(ax, "Bouquetin ibérique en Andalousie, 2017-2024", f"{len(obs):,} observations localisées à 1 km près et {len(chasse):,} mailles de 5 km avec un prélèvement de chasse déclaré".replace(",", " "))
legende(ax, [
    Line2D([], [], marker="o", linestyle="none", markerfacecolor=ROUGE, markeredgecolor="white", markersize=6, label="observation (GBIF)"),
    mpatches.Patch(facecolor=ORANGE, alpha=0.45, label="maille de 5 km avec prélèvement\n(tableaux de chasse, 2017-2021)"),
])

# Encart : place de l'Andalousie en Espagne (sans les Canaries)
espagne = gpd.read_file(GADM1).to_crs(zone.crs)
espagne = espagne[espagne.representative_point().x > -500_000]
encart = ax.inset_axes([0.0, 0.1, 0.19, 0.25])
encart.set_zorder(30)
espagne.plot(ax=encart, color="#d9d9d9", edgecolor="white", linewidth=0.3)
espagne[espagne["NAME_1"].str.contains("Andaluc", na=False)].plot(ax=encart, color=ROUGE)
encart.set_xlim(-50_000, 1_150_000)
encart.set_ylim(3_950_000, 4_900_000)
encart.set_xticks([])
encart.set_yticks([])
encart.set_xlabel("")
encart.set_ylabel("")
encart.set_title("Espagne", fontsize=7.5, color="#555555", pad=2)
for cote in encart.spines.values():
    cote.set_visible(True)
    cote.set_color("#999999")
    cote.set_linewidth(0.6)
encart.set_facecolor("white")
enregistrer(fig, "fig01_situation.png", "Sources : GBIF (téléchargement 10/12/2025), Copernicus DEM GLO-30, GADM 4.1. ETRS89 / UTM 30N.")

# ---------------------------------------------------------------------------
# 2. Évolution annuelle
# ---------------------------------------------------------------------------
annuel = pd.read_csv(os.path.join(RESULTATS, "dhi_annuel.csv"))
fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), sharex=True)
for ax, (col, titre, unite) in zip(axes, (("P", "P, productivité cumulée", "NDVI cumulé"), ("M", "M, minimum annuel", "NDVI"), ("V", "V, variabilité saisonnière", "coefficient de variation"), ("DHI", "DHI synthétique", "sans unité"))):
    ax.plot(annuel["annee"], annuel[col], color="#2166ac", linewidth=1.6, marker="o", markersize=4, markerfacecolor="white", zorder=3)
    ax.set_title(titre, loc="left", fontsize=9.5)
    ax.set_ylabel(unite)
    ax.set_xticks(annuel["annee"][::2])
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    marge_y = (annuel[col].max() - annuel[col].min()) * 0.6
    ax.set_ylim(annuel[col].min() - marge_y, annuel[col].max() + marge_y)
ax2 = axes[3].twinx()
ax2.bar(annuel["annee"], annuel["observations"], color=ROUGE, alpha=0.18, width=0.7, zorder=0)
ax2.set_ylabel("observations GBIF localisées", color=ROUGE)
ax2.tick_params(axis="y", colors=ROUGE)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(ROUGE)
axes[3].set_zorder(ax2.get_zorder() + 1)
axes[3].patch.set_visible(False)
tau, p = synthese["tendance_kendall_tau"], synthese["tendance_kendall_p"]
fig.suptitle(f"Moyennes régionales des composantes du DHI, 2017-2024 : pas de tendance (DHI : tau de Kendall = {tau:.2f}, p = {p:.2f})".replace(".", ","), x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
enregistrer(fig, "fig02_dhi_annuel.png", "Sources : Copernicus Sentinel-2 (Earth Engine), GBIF. Composites mensuels médians, nuages masqués par la bande SCL.")

# ---------------------------------------------------------------------------
# 3. DHI moyen et variation 2017-2024
# ---------------------------------------------------------------------------
dhi = raster("DHI_moyen")
fig, ax = nouvelle_carte()
fond_sans_donnee(ax)
im = ax.imshow(dhi, cmap="YlGn", extent=etendue_raster, vmin=0.1, vmax=0.9, interpolation="nearest", zorder=1)
fond_relief(ax, alpha=0.18)
habiller(ax, "Qualité de l'habitat : DHI moyen 2017-2024", "0,5 P + 0,3 M + 0,2 (1 - V), composantes normalisées sur toute la période")
barre_couleur(fig, ax, im, "DHI moyen (sans unité)", ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
enregistrer(fig, "fig03a_dhi_moyen.png")

delta = raster("DHI_delta_2024_2017")
lim = float(np.nanpercentile(np.abs(delta), 98))
fig, ax = nouvelle_carte()
fond_sans_donnee(ax)
im = ax.imshow(delta, cmap="BrBG", extent=etendue_raster, norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest", zorder=1)
fond_relief(ax, alpha=0.15)
hausse, baisse = synthese["part_hausse_delta_sup_0.02"], synthese["part_baisse_delta_inf_-0.02"]
habiller(ax, "Variation du DHI entre 2017 et 2024", f"{hausse:.0%} du territoire en hausse de plus de 0,02, {baisse:.0%} en baisse de plus de 0,02 (deux années isolées, pas une tendance)".replace("%", " %"))
borne = np.floor(lim * 100) / 100
barre_couleur(fig, ax, im, "DHI 2024 - DHI 2017 (sans unité)", ticks=[-borne, -borne / 2, 0, borne / 2, borne], format_="{x:+.2f}")
enregistrer(fig, "fig03b_dhi_variation.png")

# ---------------------------------------------------------------------------
# 4. Composite coloré des trois composantes, comme dans Radeloff et al. (2019)
# ---------------------------------------------------------------------------
comp = {c: raster(f"{c}_moyen") for c in ("P", "M", "V")}


def etirer(arr):
    lo, hi = np.nanpercentile(arr, [2, 98])
    return np.clip((arr - lo) / (hi - lo), 0, 1)


rvb = np.dstack([etirer(comp["P"]), etirer(comp["M"]), etirer(comp["V"])])
rvb = np.where(np.isfinite(rvb).all(axis=-1, keepdims=True), rvb, np.nan)
alpha = np.where(np.isfinite(rvb[..., 0]), 1.0, 0.0)
fig, ax = nouvelle_carte()
fond_sans_donnee(ax)
ax.imshow(np.dstack([np.nan_to_num(rvb, nan=1), alpha]), extent=etendue_raster, interpolation="nearest", zorder=1)
habiller(ax, "Les trois composantes du DHI, 2017-2024", "Rouge = productivité cumulée (P), vert = productivité minimale (M), bleu = variabilité saisonnière (V)")
legende(ax, [
    mpatches.Patch(color="#ffff00", label="jaune : productif toute l'année (P et M forts)"),
    mpatches.Patch(color="#ff00ff", label="magenta : productif mais très saisonnier"),
    mpatches.Patch(color="#3030ff", label="bleu : peu productif, très saisonnier"),
    mpatches.Patch(color="#202020", label="sombre : peu productif et stable"),
    mpatches.Patch(color="#c8c8c8", label="sans donnée (nuages, neige, eau)"),
], titre="Lecture des couleurs")
enregistrer(fig, "fig04_composantes_rvb.png")

# ---------------------------------------------------------------------------
# 5. Utilisation et disponibilité : résumé par AUC, puis distributions
# ---------------------------------------------------------------------------
ua1 = pd.read_csv(os.path.join(RESULTATS, "utilisation_disponibilite.csv")).set_index("variable")
ua5 = pd.read_csv(os.path.join(RESULTATS, "utilisation_disponibilite_chasse_5km.csv")).set_index("variable")
ordre = ["P_moyen", "M_moyen", "V_moyen", "DHI_moyen", "altitude", "pente", "dist_routes"]
hyp = {"P_moyen": "H1", "M_moyen": "H2", "V_moyen": "H3"}
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.axvspan(0.5, 1, color="#f4f9f4", zorder=0)
ax.axvspan(0, 0.5, color="#fbf4f2", zorder=0)
ax.axvline(0.5, color=GRIS, linewidth=1, zorder=1)
for i, v in enumerate(ordre):
    y = len(ordre) - 1 - i
    a1, a5 = ua1.loc[v, "auc"], ua5.loc[v, "auc"]
    ax.plot([a1, a5], [y, y], color="#bbbbbb", linewidth=1, zorder=2)
    ax.scatter(a1, y, s=55, color=ROUGE, zorder=3, label="observations, mailles de 1 km" if i == 0 else None)
    ax.scatter(a5, y, s=55, color=ORANGE, marker="s", zorder=3, label="tableaux de chasse, mailles de 5 km" if i == 0 else None)
ax.set_yticks(range(len(ordre)))
ax.set_yticklabels([f"{LIBELLES[v]} ({hyp[v]})" if v in hyp else LIBELLES[v] for v in ordre][::-1])
ax.set_xlim(0.2, 0.9)
ax.set_xlabel("AUC : probabilité qu'une maille occupée ait une valeur plus forte qu'une maille quelconque")
ax.text(0.52, len(ordre) - 0.45, "à droite, valeurs plus fortes là où vit l'espèce", fontsize=8, color="#3a7d44")
ax.text(0.48, len(ordre) - 0.45, "← valeurs plus faibles", fontsize=8, color="#a0522d", ha="right")
ax.set_ylim(-0.6, len(ordre) - 0.2)
ax.legend(loc="lower left", fontsize=8)
ax.set_title("Ce qui distingue les zones occupées du reste de l'Andalousie", loc="left")
fig.tight_layout(rect=(0, 0.03, 1, 1))
enregistrer(fig, "fig05a_selection_auc.png", "AUC = U de Mann-Whitney / (n1 x n2). 0,5 = aucune différence avec le territoire disponible.")

ech = pd.read_csv(os.path.join(SORTIES, "echantillons.csv"))
fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
for ax, v in zip(axes.ravel(), ["P_moyen", "M_moyen", "V_moyen", "altitude", "pente", "dist_routes"]):
    d = ech[ech["variable"] == v]
    facteur = 0.001 if v == "dist_routes" else 1
    groupes = [d[(d["echelle"] == e) & (d["groupe"] == g)]["valeur"] * facteur for e in ("1 km", "5 km") for g in ("territoire", "presence")]
    bp = ax.boxplot(groupes, positions=[0, 0.8, 2, 2.8], widths=0.6, showfliers=False, patch_artist=True, medianprops={"color": "black", "linewidth": 1.2})
    for patch, couleur in zip(bp["boxes"], ["#d9d9d9", ROUGE, "#d9d9d9", ORANGE]):
        patch.set_facecolor(couleur)
        patch.set_edgecolor("#555555")
    ax.set_xticks([0.4, 2.4], ["1 km", "5 km"])
    ax.set_title(LIBELLES[v], loc="left", fontsize=9)
    ax.set_ylabel(UNITES[v])
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
fig.legend(handles=[mpatches.Patch(facecolor="#d9d9d9", edgecolor="#555555", label="territoire disponible"), mpatches.Patch(facecolor=ROUGE, label="mailles de 1 km avec observation"), mpatches.Patch(facecolor=ORANGE, label="mailles de 5 km avec prélèvement")], loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=3, fontsize=8)
fig.suptitle("Conditions dans les zones occupées et sur le territoire", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0.03, 1, 0.91))
enregistrer(fig, "fig05b_distributions.png", "Boîtes : quartiles et médiane ; moustaches : 1,5 écart interquartile. Sources : Sentinel-2, Copernicus DEM, OpenStreetMap, GBIF.")

# ---------------------------------------------------------------------------
# 6. Grille de 20 km et points chauds
# ---------------------------------------------------------------------------
fig, ax = nouvelle_carte()
fond_relief(ax, alpha=0.35)
bornes = [0, 1, 6, 21, 101, 10_000]
couleurs = ["#f7f7f7", "#fdd49e", "#fc8d59", "#d7301f", "#7f0000"]
classes = np.digitize(g20["nb_obs"], bornes[1:-1])
g20.assign(classe=classes).plot(ax=ax, column="classe", cmap=ListedColormap(couleurs), norm=BoundaryNorm(np.arange(-0.5, 5.5), 5), edgecolor="white", linewidth=0.4, alpha=0.9, zorder=2)
habiller(ax, "Observations par cellule de 20 km", f"I de Moran (log du nombre d'observations) = {synthese['moran_grille_observations_I']:.2f}, p = {synthese['moran_grille_observations_p']:.3f}".replace(".", ","), limites=True)
legende(ax, [mpatches.Patch(facecolor=c, edgecolor="#bbbbbb", label=l) for c, l in zip(couleurs, ["aucune", "1 à 5", "6 à 20", "21 à 100", "plus de 100"])], titre="observations")
enregistrer(fig, "fig06a_grille_20km.png", "Sources : GBIF (observations localisées à 1 km près), GADM 4.1. ETRS89 / UTM 30N.")

fig, ax = nouvelle_carte()
fond_relief(ax, alpha=0.8)
g5[g5["point_chaud"]].plot(ax=ax, facecolor=ROUGE, edgecolor="none", alpha=0.75, zorder=3)
noyaux.boundary.plot(ax=ax, color="black", linewidth=1.1, zorder=4)
for i, n in noyaux.iterrows():
    pt = n.geometry.representative_point()
    ax.text(pt.x, pt.y + 14_000, f"{i + 1}", fontsize=9, fontweight="bold", ha="center", va="center", zorder=6, bbox={"boxstyle": "circle,pad=0.25", "facecolor": "white", "edgecolor": "black", "linewidth": 0.8})
part = synthese["part_obs_en_point_chaud"]
habiller(ax, "Points chauds Getis-Ord Gi* du nombre d'observations", f"Mailles de 5 km, voisinage de 10 km, 999 permutations ; {part:.0%} des observations tombent dans un point chaud".replace("%", " %"))
legende(ax, [mpatches.Patch(facecolor=ROUGE, alpha=0.75, label="point chaud (z > 1,96, p < 0,05)"), mpatches.Patch(facecolor="none", edgecolor="black", label="noyau de population (numéroté)")])
enregistrer(fig, "fig06b_points_chauds.png", "Sources : GBIF (observations localisées à 1 km près), Copernicus DEM (relief). Grille européenne EPSG:3035.")

# ---------------------------------------------------------------------------
# 7. Distance aux routes
# ---------------------------------------------------------------------------
routes = pd.read_csv(os.path.join(RESULTATS, "routes.csv"))
d1 = ech[(ech["variable"] == "dist_routes") & (ech["echelle"] == "1 km")]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for groupe, libelle, couleur in (("territoire", "territoire andalou (10 000 points tirés au hasard)", GRIS), ("presence", "mailles de 1 km avec observation", ROUGE)):
    v = np.sort(d1[d1["groupe"] == groupe]["valeur"].to_numpy()) / 1000
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=couleur, linewidth=2, label=libelle)
r5 = routes.set_index("seuil_m").loc[5000]
ax.text(5.15, 0.99, "5 km", fontsize=7.5, color="#888888", va="top")
ax.axvline(5, color="#bbbbbb", linestyle="--", linewidth=1)
ax.scatter([5, 5], [r5["part_territoire"], r5["part_mailles_1km"]], color=[GRIS, ROUGE], s=25, zorder=4)
ax.text(4.6, r5["part_territoire"] + 0.06, f"{r5['part_territoire']:.0%} du territoire".replace("%", " %"), fontsize=8, color="#555555", ha="right", va="center")
ax.text(5.4, r5["part_mailles_1km"] - 0.03, f"{r5['part_mailles_1km']:.0%} des mailles occupées".replace("%", " %"), fontsize=8, color=ROUGE, ha="left", va="top")
ax.set_xlim(0, 35)
ax.set_ylim(0, 1.01)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, decimals=0, symbol=" %"))
ax.set_xlabel("distance à la route principale la plus proche (km)")
ax.set_ylabel("part à moins de cette distance")
ax.grid(color="#eeeeee", linewidth=0.6)
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Le bouquetin est un peu plus loin des routes que la moyenne du territoire", loc="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0.04, 1, 1))
enregistrer(fig, "fig07_routes.png", "Routes principales OpenStreetMap : motorway, trunk, primary. Distances euclidiennes sur grille de 250 m.")

# ---------------------------------------------------------------------------
# 8. Forêt aléatoire : importance des variables
# ---------------------------------------------------------------------------
imp = pd.read_csv(os.path.join(RESULTATS, "random_forest_importance.csv")).sort_values("importance_moyenne")
fig, ax = plt.subplots(figsize=(7.5, 3.8))
couleurs = ["#2166ac" if v in ("altitude", "pente") else ("#5aae61" if v.endswith("_moyen") else GRIS) for v in imp["variable"]]
ax.barh([LIBELLES[v] for v in imp["variable"]], imp["importance_moyenne"], xerr=imp["importance_ecart_type"], color=couleurs, error_kw={"ecolor": "#555555", "linewidth": 0.8, "capsize": 2})
ax.axvline(0, color="#555555", linewidth=0.8)
ax.set_xlabel("perte d'AUC quand la variable est mélangée (moyenne des 5 plis)")
ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
auc_s, auc_a = synthese["rf_auc_spatial_moyenne"], synthese["rf_auc_aleatoire_moyenne"]
ax.set_title(f"Forêt aléatoire : le relief explique la présence, pas la végétation\nAUC = {auc_s:.2f} en validation par blocs spatiaux, {auc_a:.2f} en découpage aléatoire".replace(".", ","), loc="left", fontsize=10)
ax.legend(handles=[mpatches.Patch(color="#2166ac", label="relief"), mpatches.Patch(color="#5aae61", label="composantes du DHI"), mpatches.Patch(color=GRIS, label="routes")], loc="lower right", fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1))
enregistrer(fig, "fig08_foret_importance.png", "Validation croisée par blocs de 50 km, 5 plis. Présences : une par maille de 1 km ; 10 000 points de fond.")

# ---------------------------------------------------------------------------
# 9. Probabilité de présence et zones favorables inoccupées
# ---------------------------------------------------------------------------
proba = raster("probabilite_presence")
fig, ax = nouvelle_carte()
fond_sans_donnee(ax)
im = ax.imshow(proba, cmap="magma", extent=etendue_raster, vmin=0, vmax=1, interpolation="nearest", zorder=1)
fond_relief(ax, alpha=0.15)
habiller(ax, "Probabilité de présence prédite par la forêt aléatoire", "Prédicteurs : pente, altitude, composantes P, M, V du DHI, distance aux routes ; grille de 250 m")
barre_couleur(fig, ax, im, "probabilité de présence", ticks=[0, 0.25, 0.5, 0.75, 1])
enregistrer(fig, "fig09a_probabilite_presence.png")

seuil = synthese["rf_seuil_p10"]
recol = raster("potentiel_recolonisation")
classes = np.full(dedans.shape, np.nan)
classes[np.isfinite(proba)] = 0
classes[proba >= seuil] = 1
classes[recol == 1] = 2
fig, ax = nouvelle_carte()
couleurs = ["#f0f0f0", "#9ecae1", "#08519c"]
ax.imshow(classes, cmap=ListedColormap(couleurs), extent=etendue_raster, vmin=-0.5, vmax=2.5, interpolation="nearest", zorder=1)
fond_relief(ax, alpha=0.2)
obs.plot(ax=ax, color=ROUGE, markersize=1.5, zorder=4)
surface = synthese["surface_favorable_non_occupee_km2"]
surface_txt = f"{surface:,.0f}".replace(",", " ")
seuil_txt = f"{seuil:.2f}".replace(".", ",")
habiller(ax, "Habitat favorable sans observation à moins de 10 km", f"Favorable : probabilité ≥ {seuil_txt} (seuil P10, dépassé par 90 % des présences prédites hors de leur bloc) ; {surface_txt} km² sans observation proche")
legende(ax, [
    mpatches.Patch(facecolor=couleurs[0], edgecolor="#bbbbbb", label="défavorable"),
    mpatches.Patch(facecolor=couleurs[1], label="favorable, observations à moins de 10 km"),
    mpatches.Patch(facecolor=couleurs[2], label="favorable, aucune observation à moins de 10 km"),
    Line2D([], [], marker="o", linestyle="none", color=ROUGE, markersize=3, label="observation"),
])
enregistrer(fig, "fig09b_habitat_favorable.png")

# ---------------------------------------------------------------------------
# 10. Corridors
# ---------------------------------------------------------------------------
resistance = raster("resistance")
fig, ax = nouvelle_carte()
fond_sans_donnee(ax)
im = ax.imshow(resistance, cmap="Greys", extent=etendue_raster, vmin=1, vmax=float(np.nanpercentile(resistance, 99)), interpolation="nearest", zorder=1)
corridors.plot(ax=ax, color="white", linewidth=3.2, zorder=4)
corridors.plot(ax=ax, column="cout_par_km", cmap="viridis_r", linewidth=1.8, zorder=5)
noyaux.plot(ax=ax, facecolor=ROUGE, edgecolor="black", alpha=0.7, linewidth=0.8, zorder=6)
for i, n in noyaux.iterrows():
    pt = n.geometry.representative_point()
    ax.text(pt.x, pt.y + 14_000, f"{i + 1}", fontsize=9, fontweight="bold", ha="center", va="center", zorder=7, bbox={"boxstyle": "circle,pad=0.25", "facecolor": "white", "edgecolor": "black", "linewidth": 0.8})
habiller(ax, "Chemins de moindre coût entre les noyaux de population", "Résistance = 1 + 4 (1 - DHI) + 5 exp(- distance aux routes / 500 m) ; noyaux issus des points chauds Gi*")
barre_couleur(fig, ax, im, "résistance au déplacement (sans unité)", sans_donnee=False)
ax.legend(handles=[Line2D([], [], color="#440154", linewidth=2, label="chemin coûteux"), Line2D([], [], color="#fde725", linewidth=2, label="chemin peu coûteux"), mpatches.Patch(facecolor=ROUGE, alpha=0.7, edgecolor="black", label="noyau de population")], loc="lower right", bbox_to_anchor=(0.98, 0.16), fontsize=8)
enregistrer(fig, "fig10_corridors.png")

print(f"Étape 4 terminée : {FIGURES}")
