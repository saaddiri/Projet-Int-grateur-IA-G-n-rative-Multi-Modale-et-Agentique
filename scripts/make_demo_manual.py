"""Génère un manuel de maintenance FICTIF du compresseur CMP-101 (document de démonstration).

Il relie la documentation indexée aux équipements simulés du serveur MCP et sert à
démontrer le filtrage des documents selon les autorisations (réservé aux techniciens et
superviseurs de l'atelier A, voir data/documents_acl.json).

Usage :  python -m scripts.make_demo_manual   ->  data/manuel_CMP-101.pdf
"""

from __future__ import annotations

import pymupdf

from maintidoc.config import ROOT_DIR

OUTPUT = ROOT_DIR / "data" / "manuel_CMP-101.pdf"

PAGES = [
    ("Manuel de maintenance — Compresseur d'air à vis CMP-101 (document fictif de démonstration)", [
        "Ce manuel décrit les opérations de maintenance de premier et second niveau du compresseur d'air "
        "à vis CMP-101 (37 kW) installé dans l'atelier A. Il est réservé au personnel habilité. "
        "Toute intervention se fait machine consignée, sauf les contrôles en fonctionnement explicitement "
        "indiqués. Ce document est fictif et sert uniquement à la démonstration du projet MaintiDoc.",
        "1. Vibrations anormales au démarrage. Causes probables, par ordre de fréquence : desserrage des "
        "silentblocs ou des boulons d'ancrage du châssis ; usure de l'accouplement élastique entre le moteur "
        "et le bloc vis ; défaut de roulement du moteur électrique ; démarrage étoile-triangle trop court. "
        "Contrôles recommandés : vérifier le serrage des silentblocs au couple prescrit, inspecter "
        "visuellement les éléments élastiques de l'accouplement, mesurer le niveau vibratoire au palier "
        "moteur et le comparer à la valeur de référence relevée à la mise en service. Si les vibrations "
        "persistent après resserrage, ouvrir un ticket pour une analyse vibratoire par le service méthodes.",
    ]),
    ("2. Alarme température d'huile élevée", [
        "L'alarme se déclenche lorsque la température en sortie de bloc vis dépasse le seuil réglé dans le "
        "contrôleur. Causes probables : radiateur d'huile encrassé côté air, ventilation de la salle "
        "insuffisante, niveau d'huile bas, vanne thermostatique bloquée. Actions : nettoyer le radiateur à "
        "l'air comprimé de l'intérieur vers l'extérieur, contrôler le niveau d'huile à l'arrêt après "
        "décompression, vérifier que les grilles de ventilation de la salle ne sont pas obstruées. Une "
        "alarme qui revient en moins d'une semaine après nettoyage justifie le contrôle de la vanne "
        "thermostatique par un technicien de second niveau.",
        "3. Purge des condensats. Le purgeur électronique doit être testé chaque semaine avec le bouton de "
        "test. Un purgeur bloqué provoque la présence d'eau dans le réseau d'air comprimé.",
    ]),
]

TABLE_TITLE = "Tableau 1 : Contrôles préventifs du compresseur CMP-101"
TABLE = [
    ["Opération", "Périodicité", "Niveau"],
    ["Contrôle du niveau d'huile", "Hebdomadaire", "1"],
    ["Test du purgeur de condensats", "Hebdomadaire", "1"],
    ["Nettoyage du radiateur d'huile", "500 h", "1"],
    ["Serrage des silentblocs et ancrages", "2000 h", "2"],
    ["Remplacement du séparateur d'huile", "4000 h", "2"],
    ["Analyse vibratoire du bloc vis", "8000 h", "2"],
]


def build() -> None:
    doc = pymupdf.open()
    for title, paragraphs in PAGES:
        page = doc.new_page()
        y = 60
        page.insert_textbox(pymupdf.Rect(50, y, 545, y + 50), title, fontsize=14, fontname="helv")
        y += 60
        for paragraph in paragraphs:
            rect = pymupdf.Rect(50, y, 545, y + 220)
            page.insert_textbox(rect, paragraph, fontsize=10.5, fontname="helv")
            y += 20 + 14 * (len(paragraph) // 85 + 1)
    page = doc.new_page()
    page.insert_text((50, 70), TABLE_TITLE, fontsize=12, fontname="helv")
    x_cols, row_h, top = [50, 300, 430, 545], 24, 90
    for r, row in enumerate(TABLE):
        y = top + r * row_h
        for c, cell in enumerate(row):
            page.draw_rect(pymupdf.Rect(x_cols[c], y, x_cols[c + 1], y + row_h), color=(0, 0, 0), width=0.6)
            page.insert_text((x_cols[c] + 5, y + 16), cell, fontsize=10, fontname="helv")
    doc.save(OUTPUT)
    print(f"Écrit : {OUTPUT}")


if __name__ == "__main__":
    build()
