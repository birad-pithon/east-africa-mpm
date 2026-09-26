# Verified mineral-deposit reference study

Generated: 2026-09-24T13:11:56.798759+00:00

## Scope and evidence rule

Source: `D:\Users\HP\COC AI\east-africa-mpm\data\raw\usgs_mrds\mrds-trim.shp` (USGS MRDS global extract).

The verified subset contains every record classified by MRDS as `Producer` or `Past Producer` and carrying a target commodity code. This is a mining-status evidence class; it does not independently verify grade, tonnage, ownership, or current operation. MRDS coverage outside the United States is incomplete.

Prospects, occurrences, unknown-status records, and processing plants are retained in the source but are not called verified deposits. Reference records are not written to training labels and are marked `label_eligible=False`.

## Scientific transfer framework

A shared commodity does not establish a shared genesis. Sn-W-Ta is evaluated against LCT-pegmatite and granite-related greisen models; Cu-Zn is separated into sediment-hosted, fault-carbonate, and VMS concepts; bauxite is evaluated as a weathering-profile problem on plateau surfaces. The source extract has no host-rock, age, deposit-type, alteration, ore-mineral, reserve, or tonnage fields, so individual records cannot be assigned to a single model from this file alone.

Each point is related geographically to every same-group permissive target window. The configured target's deposit model and spectral, terrain, and geophysics signatures provide the scientific interpretation; distance alone does not establish geological equivalence.

## Geographic relation to target regions

`inside_permissive_window` means only that a point lies inside a configured search window. `within_100_km_transfer_context` identifies geographic proximity for local review. Both require deposit-model validation; neither asserts mineralisation or creates a target.

| Target | Group | Confidence | Relations | Inside window | Within 100 km |
|---|---|---|---:|---:|---:|
| `kab_burundi_north` | tin_tungsten_tantalum | high | 5,383 | 0 | 3 |
| `kab_tanzania_karagwe` | tin_tungsten_tantalum | high | 5,383 | 0 | 3 |
| `kab_uganda_southwest` | tin_tungsten_tantalum | high | 5,383 | 1 | 5 |
| `kibaran_kigoma` | tin_tungsten_tantalum | moderate | 5,383 | 0 | 0 |
| `kibaran_ubende_mpanda` | tin_tungsten_tantalum | low | 5,383 | 0 | 0 |
| `lufilian_kolwezi_kambove` | copper_zinc | high | 31,078 | 18 | 6 |
| `lufilian_likasi_tenke` | copper_zinc | high | 31,078 | 8 | 15 |
| `zambia_copperbelt_north` | copper_zinc | high | 31,078 | 13 | 7 |
| `kibaran_vms_rwanda_west` | copper_zinc | moderate | 31,078 | 0 | 0 |
| `arabian_nubian_ethiopia_west` | copper_zinc | moderate | 31,078 | 0 | 0 |
| `ubendian_kate_kipili` | copper_zinc | low | 31,078 | 0 | 0 |
| `usambara_east_extension` | bauxite | high | 1,113 | 0 | 0 |
| `southern_highlands_mbeya` | bauxite | moderate | 1,113 | 0 | 0 |
| `kenya_coast_shimba_kwale` | bauxite | moderate | 1,113 | 0 | 0 |
| `ethiopia_southwest_bench` | bauxite | moderate | 1,113 | 0 | 0 |
| `uganda_busoga_plateau` | bauxite | low | 1,113 | 0 | 0 |
| `southsudan_equatoria_highlands` | bauxite | low | 1,113 | 0 | 0 |

## Inventory

- Target-commodity records available: **70,827**
- Producer/Past Producer reference records: **36,122**
- Unique reference site names: **25,484**
- Target-transfer relations: **220,061**

| Mineral group | MRDS codes | Records | Sites | Current | Past |
|---|---|---:|---:|---:|---:|
| tin_tungsten_tantalum | SN / TA / W | 5,383 | 4,615 | 1,412 | 3,971 |
| copper_zinc | CU / ZN | 31,078 | 21,713 | 4,887 | 26,191 |
| bauxite | AL | 1,113 | 921 | 341 | 772 |

Multi-commodity records are represented in every group they match, so group counts do not sum to unique MRDS IDs.

## Use and limitations

1. Use the inventory to widen geological priors and select external review areas.
2. Build separate strata or features for Cu-Zn VMS, carbonate-replacement, and sediment-hosted signatures.
3. Validate source status, commodity association, host lithology, alteration, age, and ore mineralogy before promoting a record to a label or a target.
4. Validate against the target's common feature grid before any transfer-learning or domain-adaptation experiment.

The accompanying GeoPackage contains the point records and transfer relations; the JSON summary contains machine-readable counts and provenance.
The nearest curated world anchor in the point layer is a navigation aid for literature review, not a classification of the MRDS record's genesis. Host rock, age, alteration, ore mineralogy, grade, and tonnage require record-level geological verification.
