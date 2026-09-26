"""Curated seed deposits for positive-label generation.

Coordinates are literature-derived WGS-84 points, accurate to roughly
1-3 km ('approx' precision) - sufficient as training seeds; refine with
national survey data when it arrives (see data_sources.yaml layer 1).

References (deposit-genesis literature):
  * Rwanda Sn-W-Ta: Dewaele et al. (2015-2016), KAB granite-related
    mineralisation (Nyakabingo W-veins; Gatumba Nb-Ta-Sn pegmatites;
    Musha-Ntunga Sn-W veins).
  * Kipushi: Kamona & Friedrich (2007) - Cu-Zn-Pb, Lufilian arc.
  * Kamoa-Kakula: Schuh et al. (2012); Broughton (2014) - basin-hosted Cu.
  * Usambara bauxite: Tanzania GSD laterite reports.
"""
from __future__ import annotations

# Each seed: name, country, lon/lat (WGS-84), mrds commodity tokens,
# belt id (configs/belts.yaml), precision + reference provenance.
SEED_DEPOSITS: list[dict] = [
    # ── Rwanda — Karagwe-Ankole Belt (Sn-W-Ta) ──────────────────────────
    {
        "name": "Nyakabingo",
        "country": "Rwanda",
        "lon": 30.067, "lat": -1.917,
        "commodities": "W SN",
        "belt": "karagwe_ankole",
        "precision": "approx",
        "reference": "Dewaele et al. 2016; Rulindo District W-vein field",
    },
    {
        "name": "Musha",
        "country": "Rwanda",
        "lon": 30.300, "lat": -1.950,
        "commodities": "SN W",
        "belt": "karagwe_ankole",
        "precision": "approx",
        "reference": "Dewaele et al. 2015; Rwamagana District Sn-W veins",
    },
    {
        "name": "Ntunga",
        "country": "Rwanda",
        "lon": 30.350, "lat": -2.017,
        "commodities": "SN W TA",
        "belt": "karagwe_ankole",
        "precision": "approx",
        "reference": "Dewaele et al. 2015; Musha-Ntunga vein swarm",
    },
    {
        "name": "Gatumba",
        "country": "Rwanda",
        "lon": 29.640, "lat": -1.994,
        "commodities": "SN TA W",
        "belt": "karagwe_ankole",
        "precision": "approx",
        "reference": "Dewaele et al. 2016; Ngororero Nb-Ta-Sn pegmatites",
    },
    # ── DRC — Central African Copperbelt (Cu-Zn) ────────────────────────
    {
        "name": "Kipushi",
        "country": "DR Congo",
        "lon": 27.233, "lat": -11.733,
        "commodities": "ZN CU PB",
        "belt": "central_african_copperbelt",
        "precision": "known",
        "reference": "Kamona & Friedrich 2007; high-grade Zn-Cu-Pb mine",
    },
    {
        "name": "Kamoa-Kakula",
        "country": "DR Congo",
        "lon": 24.100, "lat": -10.600,
        "commodities": "CU",
        "belt": "central_african_copperbelt",
        "precision": "approx",
        "reference": "Broughton 2014; Ivanhoe Mines Kakula complex",
    },
    # ── Tanzania — Usambara lateritic bauxite ───────────────────────────
    {
        "name": "Lushoto",
        "country": "Tanzania",
        "lon": 38.302, "lat": -4.793,
        "commodities": "AL",
        "belt": "usambara_bauxite",
        "precision": "known",
        "reference": "Tanzania GSD; Usambara lateritic plateaus",
    },
    {
        "name": "Magamba",
        "country": "Tanzania",
        "lon": 38.267, "lat": -4.767,
        "commodities": "AL",
        "belt": "usambara_bauxite",
        "precision": "approx",
        "reference": "Magamba area, West Usambara laterite plateau",
    },
]
