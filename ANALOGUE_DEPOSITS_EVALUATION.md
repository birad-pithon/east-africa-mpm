# World-deposit analogue evaluation → East Africa scene selection

**Date:** 2026-09-22
**Scope:** transfer deposit-model knowledge from world-class Sn-W-Ta,
Cu-Zn and bauxite districts onto permissive ground in the 7-country
region (Rwanda, Uganda, Kenya, Tanzania, Burundi, South Sudan, Ethiopia).
**Machine-readable windows:** `configs/analogue_targets.yaml` (17 targets).

> **Methodological guardrail.** Analogue deposits are *never* added as
> positive labels: they sit on different feature grids (domain shift) and
> would corrupt spatial CV. They inform *where to look* (scene
> selection) and *what signature to expect* (fingerprint design). Labels
> come only from USGS MRDS in-belt filtering + national surveys + field
> validation (`src.labels.build_labels`).

## 1. Sn-W-Ta — LCT pegmatite / greisen system

**World analogues evaluated.**
Manono-Kitotolo (DRC, same Kibaran orogen — zoned LCT pegmatite,
spodumene/cassiterite/coltan/lepidolite/tourmaline, ~940 Ma G4 cupolas
in metapelites); Greenbushes (Australia — giant zoned LCT pegmatite);
Bikita (Zimbabwe — LCT pegmatite in craton margin); Erzgebirge
greisen (Altenberg/Cinovec — muscovite-topaz-quartz greisen above
evolved granite cupolas); Jos Plateau (Nigeria — granite-related Sn).

**Transferable criteria (geology).**
(1) Peraluminous S-type granite cupolas of the late (G4-type) generation
intruding Kibaran metapelites; (2) pegmatite-swarm / vein corridors
within ~2 km of contacts (fingerprint `geology_contact` half-decay
2.0 km, w=0.75); (3) high fault/shear density as fluid pathways;
(4) fractionation indicators (Li-Rb-Cs-Ta enrichment toward cupola).

**Transferable criteria (spectra).**
Al-OH 2.20 um (muscovite/lepidolite/sericite) primary;
Mg-OH/CO3 2.33-2.35 um greisen-carbonate halo secondary;
Fe-oxide 0.86-0.90 um gossan screen. Sentinel-2 B11/B12 gives only a
coarse `aloh_ratio` screen; muscovite-vs-lepidolite-vs-carbonate
discrimination *requires* EnMAP probes [2.20, 2.35] um.

**East Africa scenes (5):**
kab_burundi_north (Burundi, high — along-strike KAB continuation);
kab_tanzania_karagwe (Tanzania, high — Karagwe district);
kab_uganda_southwest (Uganda, high — SW-Uganda KAB corridor);
kibaran_kigoma (Tanzania, moderate — needs GST chemistry check);
kibaran_ubende_mpanda (Tanzania, low — structural maps first).

## 2. Cu-Zn — sediment-hosted stratiform + Kipushi + VMS systems

**World analogues evaluated.**
Zambian Copperbelt (Nkana-Mufulira-Nchanga-Konkola — ore-shale at the
basement-Roan contact, redox trap); Congolese Copperbelt
(Tenke-Fungurume/Kolwezi/Likasi — Mines Subgroup ore shale, dilational
fold hinges); Kamoa-Kakula (external-belt stratiform Cu); Kipushi
(fault-carbonate Zn-Cu breccia, dolomitization + silicification +
gossan); Kupferschiefer (reduced black shale over red beds —
redox-front process model); Kidd Creek/Noranda (bimodal VMS with
sericite-core / chlorite-footwall / gossan-cap zonation);
Bisha (Arabian-Nubian VMS — direct shield analogue for W Ethiopia).

**Transferable criteria (geology).**
Sediment-hosted: (1) reduced ore-shale horizon near basement or
basin-margin redox front; (2) basin architecture (magnetic/gravity);
(3) fold-hinge / fault dilational traps; (4) dolomitization +
silicification halos. Kipushi-style: fault x carbonate-panel
intersections with breccia. VMS: bimodal volcanic centres with
synvolcanic faults (a *separate* fingerprint — Cu-Zn hosts two
deposit models; do not mix their priors).

**Transferable criteria (spectra).**
Phyllic sericite 2.20 um + propylitic chlorite/epidote/carbonate
2.33 um + gossan Fe-oxide 0.86 um zonation. EnMAP probes
[2.20, 2.33, 0.86] um discriminate sericite/kaolinite/chlorite/epidote.

**East Africa scenes (6):**
lufilian_kolwezi_kambove (DRC, high — between both Cu seeds);
lufilian_likasi_tenke (DRC, high — type ore-shale ground);
zambia_copperbelt_north (Zambia, high — densest ore-shale analogue);
kibaran_vms_rwanda_west (Rwanda, moderate — VMS fingerprint, RMB first);
arabian_nubian_ethiopia_west (Ethiopia, moderate — same shield as Bisha);
ubendian_kate_kipili (Tanzania, low — GST facies check first).

## 3. Bauxite — humid plateau laterite system

**World analogues evaluated.**
Weipa (Australia — coastal plateau, kaolinite-to-gibbsite profile);
Sangaredi/Boke + Kindia (Guinea — high-plateau gibbsite/boehmite);
Minim-Martap (Cameroon — plateau bauxite, closest climate analogue);
Trombetas (Brazil — cratonic-cover plateau).

**Transferable criteria.**
(1) Plateau surfaces with slope <= 5 deg (`plateau_index` high);
(2) deep weathering under high rainfall + good drainage (plateau edge,
not swamp); (3) felsic/granitic or arkosic parent preferred for
gibbsite over kaolinite; (4) profile gibbsite/boehmite over kaolinite
with hematite/goethite duricrust — Al-OH 2.20 um + Fe-oxide 0.90 um
(EnMAP probes [2.20, 0.90] um), screened regionally by `aloh_ratio` +
`iron_oxide_ratio` + MBI/BSI bareness + `plateau_index`;
(5) radiometric Th/U residual enrichment as a check. No structural
control (fingerprint deposit-distance weight only 0.75; terrain does
the work).

**East Africa scenes (6):**
usambara_east_extension (Tanzania, high — adjoins trained AOI);
southern_highlands_mbeya (Tanzania, moderate — Guinean-type plateau);
kenya_coast_shimba_kwale (Kenya, moderate — Weipa-type coastal plateau);
ethiopia_southwest_bench (Ethiopia, moderate — highest-rainfall plateaus);
uganda_busoga_plateau (Uganda, low — DGSM occurrence check first);
southsudan_equatoria_highlands (South Sudan, low — greenfield SRTM only).

## 4. Scene-acquisition workflow (all 17 targets)

1. **Confirm permissive geology** on national survey maps
   (RMB/DGSM/GST/GSK/EGS; `data_sources.yaml` layer 1) — granite
   presence/chemistry for Sn-W-Ta; stratigraphy/carbonate facies for
   Cu-Zn; parent rock + drainage for bauxite.
2. **SRTM + Sentinel-2 dry-season ingest** per target window/CRS
   (`src.ingest.srtm_dem`, `src.ingest.sentinel2_ee`); run existing
   feature code unchanged (`spectral.py`, `plateau_index`, geology
   distances) + WMS context layers (`dlr_eoc_wms`).
3. **Screen with current fingerprints**, flag anomalies, then order
   **EnMAP L2A on-demand** (Foreground Mission / EOWEB) over flagged
   cells; ingest via `src.ingest.enmap`, discriminate via
   `src.features.hyperspectral` band_depth/SAM at the probe
   wavelengths in each target entry.
4. **Labels only from MRDS/national/field sources** inside the new
   window; validate transfer with spatial CV + LOSO before any
   prospectivity claim (see `HOLDOUT_VALIDATION_REPORT.md`).
