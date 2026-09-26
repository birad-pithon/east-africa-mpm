# East Africa Mineral Prospectivity - Data Sources and Satellite Access Summary

**Date:** 2026-09-14  
**Scope:** 7-country East Africa region (Rwanda, Uganda, Kenya, Tanzania, Burundi, South Sudan, Ethiopia)

This document summarises every mineral occurrence, geological, geophysical, satellite imagery, and reference data source identified for the East Africa MPM pipeline. Sources are registered in configs/data_sources.yaml.


## 1. Regional geological maps (lithology and structural base layer)

| # | Source | Provider | Access | URL | Format | Commodities | Status |
|---|---|---|---|---|---|---|---|
| 1 | USGS World Geology WFS | USGS / GSC | download | https://mrdata.usgs.gov/services/wfs/worldgeol | WFS to GeoPackage | all | configured |
| 2 | BRGM Africa Geology 1:10M | BRGM / CGMW | download | https://geobru.brgm.fr / https://www.sigafrique.net | GIS/shapefile-zip | all | pending-url |
| 3 | Rwanda RMB Geology | Rwanda Mines Board | direct-request | https://www.rmb.gov.rw | shapefile/geodatabase | sn,w,ta | request-required |
| 4 | Tanzania GST Geology | Geological Survey of Tanzania | direct-request | https://www.gst.go.tz | shapefile/geodatabase | ba,sn,w,ta | request-required |
| 5 | DRC CAMI Geology | CAMI (DRC) | direct-request | https://cami.cd | shapefile/geodatabase | cu,zn | request-required |
| 6 | Uganda DGSM Geology | Directorate of Geological Survey and Mines | direct-request | https://dgsm.go.ug | shapefile/geodatabase | sn,w,ta | request-required |
| 7 | Kenya GSK Geology | Geological Survey of Kenya | direct-request | https://www.geologicalsurvey.go.ke | shapefile/geodatabase/PDF | cu,zn,ba,sn,w,ta,au,gemstones | request-required |
| 8 | Ethiopia EGS Geology | Ethiopian Geological Survey / MoM | direct-request | https://mines.gov.et | shapefile/geodatabase/PDF | au,cu,zn,sn,w,ta,gemstones,industrial | request-required |
| 9 | South Sudan DSS Geology | DSS / Ministry of Mining | direct-request | -- | shapefile/geodatabase/PDF | cu,zn,ba,sn,w,ta,au | request-required |
| 10 | Burundi MINEODYN Geology | Ministry of Energy and Mines (Burundi) | direct-request | -- | shapefile/geodatabase/PDF | sn,w,ta,au,cu,zn,gemstones | request-required |
| 11 | OneGeology Portal | OneGeology / CGMW | portal | https://www.onegeology.org/ | WCS/WMS/WFS/shapefile | all | pending |

**Seven-country coverage:** All 7 countries now have national survey contacts registered (Rwanda, Uganda, Tanzania, DRC, Kenya, Ethiopia, South Sudan, Burundi) plus regional/global options.


## 2. Known deposit / occurrence points (positive labels)

| # | Source | Provider | Access | URL | Format | Commodities | Status |
|---|---|---|---|---|---|---|---|
| 1 | USGS MRDS (shapefile) | USGS | download | https://mrdata.usgs.gov/mrds/mrds-trim.zip | shapefile (point) | sn,w,ta,cu,zn,ba | configured |
| 2 | USGS MRDS (WFS) | USGS | download | https://mrdata.usgs.gov/services/wfs/mrds | WFS live | sn,w,ta,cu,zn,ba | configured |
| 3 | USGS Africa Mineral Compilation | USGS | download | (pending-url) | FileGDB/shapefile-zip | all | pending-url |
| 4 | Mindat.org | Mindat.org / Hudson Institute | portal | https://www.mindat.org/ | Web table (manual geocode) | all | pending |
| 5 | African Mineral Deposits (academic) | Academic/UN publications | manual | -- | Tables/maps PDFs | au,cu,zn,sn,w,ta,ba,gemstones | manual |
| 6 | Deposit genesis papers (KAB) | Academic papers | manual | -- | Tables/geocoded refs | sn,w,ta,cu,zn | manual |

**Coverage:** USGS MRDS covers all 7 countries globally; African Mineral Compilation adds infrastructure (roads, power, permits) for the whole continent; Mindat.org enriches with species-level mineral data.


## 3. Aeromagnetic and gravity surveys (structural/intrusion detection)

| # | Source | Provider | Access | URL | Format | Commodities | Status |
|---|---|---|---|---|---|---|---|
| 1 | EMAG2v3 Magnetic Anomaly Grid | NOAA/NCEI | download | (pending-url) | GeoTIFF/ESRI ASCII | all | pending-url |
| 2 | WDMAM (World Digital Magnetic Anomaly Map) | IGPP/BGI/CGMW | download | https://wdmam.org/ | GeoTIFF/netCDF/ESRI ASCII | sn,w,ta,cu,zn | pending-url |
| 3 | WGM2012 Gravity (BGI) | Bureau Gravimetrique International | download | (pending-url) | GeoTIFF/ESRI ASCII | all | pending-url |
| 4 | Rwanda RMB aeromagnetics | RMB / RMCA (digitised archive) | direct-request | https://www.rmb.gov.rw | raster/vector anomaly lines | sn,w,ta | request-required |

**Coverage:** EMAG2 and WDMAM provide global coverage including all 7 countries at 2-5 km resolution. National aeromagnetics are request-required for Rwanda, Uganda, Tanzania, Kenya, Ethiopia, South Sudan, Burundi.


## 4. Multispectral / hyperspectral imagery (alteration mapping)

| # | Source | Provider | Access | URL | Format | Commodities | Status |
|---|---|---|---|---|---|---|---|
| 1 | Sentinel-2 (S2_SR_HARMONIZED) | ESA Copernicus / GEE | download (EE) | COPERNICUS/S2_SR_HARMONIZED | GeoTIFF (30 m aligned) | all | configured |
| 2 | Landsat 8/9 (LC08/C02/T1_L2) | USGS / EarthExplorer / GEE | portal/EE | LANDSAT/LC08/C02/T1_L2 | GeoTIFF | ba,cu,zn | pending-url |
| 3 | ASTER (EarthExplorer) | NASA / USGS | portal | USGS/ASTER | GeoTIFF | ba,cu,zn | pending-url |
| 4 | Sentinel-2 (Copernicus DSE) | ESA / Copernicus DSE | download | https://dataspace.copernicus.eu/ | SAFE/JPEG2000/COG (Phenix) | all | configured |
| 5 | Sentinel-2 (ESA SciHub) | ESA / Copernicus | portal | https://scihub.copernicus.eu/ | SAFE/JPEG2000 | all | pending |
| 6 | Sentinel-1 SAR (Copernicus DSE) | ESA / Copernicus DSE | download | https://dataspace.copernicus.eu/ | SAFE/COG | sn,w,ta,cu,zn (structural proxies) | configured |

**Coverage:** Sentinel-2 provides 10-20 m resolution multispectral imagery for all 7 countries. Landsat provides broad-scale 30 m alteration mapping. ASTER adds shortwave/thermal bands.


## 5. Digital Elevation Models (terrain features)

| # | Source | Provider | Access | URL | Format | Commodities | Status |
|---|---|---|---|---|---|---|---|
| 1 | SRTM 30 m (USGS/SRTMGL1_003) | USGS / GEE | download (EE) | USGS/SRTMGL1_003 | GeoTIFF (30 m aligned) | all | configured |
| 2 | Copernicus GLO-30 DEM | ESA / GEE | download (EE) | COPERNICUS/DEM/GLO30 | GeoTIFF (30 m aligned) | all | configured |
| 3 | ALOS PALSAR 12.5 m DEM | JAXA / NASA ASF Vertex | portal | https://vertex.daac.asf.alaska.edu/ | GeoTIFF (12.5 m) | sn,w,ta,cu,zn | configured |
| 4 | Sentinel-2 (via Earthdata) | NASA Earthdata DAAC | portal | https://earthdata.nasa.gov/ | GeoTIFF/netCDF/HDF, OGC WCS/WMS | all | configured |

**Coverage:** SRTM (30 m) and Copernicus GLO-30 (30 m) cover all 7 countries globally. ALOS PALSAR DEM provides 12.5 m for the entire region.


## 6. Stream sediment / soil geochemistry

| # | Source | Provider | Access | URL | Format | Commodities | Status |
|---|---|---|---|---|---|---|---|
| 1 | RMCA survey geochemistry (Rwanda) | RMCA / RMB | direct-request | https://www.rmb.gov.rw | rasters/CSV tables | sn,w,ta | request-required |
| 2 | Published geochemistry reports | Academic/mining reports | manual | -- | PDFs/tables | sn,w,ta,cu,zn | manual |

**Coverage:** Currently Rwanda-focused. National surveys for Uganda, Tanzania, Kenya, Ethiopia, South Sudan, Burundi to be requested.


## 7. Cadastral / licence boundaries (soft positive proxies)

| # | Source | Provider | Access | URL | Format | Commodities | Status |
|---|---|---|---|---|---|---|---|
| 1 | RMB GIMCS licences | Rwanda Mines Board | portal | https://www.rmb.gov.rw | shapefile/GeoJSON | sn,w,ta,all | request-required |
| 2 | Tanzania TMC licences | Tanzania Mining Commission | direct-request | https://www.tmc.go.tz | shapefile/GeoJSON | sn,w,ta,cu,zn,ba | request-required |
| 3 | DRC CAMI licences | CAMI (DRC) | direct-request | https://cami.cd | shapefile/GeoJSON | cu,zn | request-required |
| 4 | Uganda DGSM licences | DGSM Uganda | direct-request | https://dgsm.go.ug | shapefile/GeoJSON | sn,w,ta | request-required |
| 5 | Kenya mining licences | Ministry of Mining (Kenya) | direct-request | https://www.geologicalsurvey.go.ke | shapefile/GeoJSON | cu,zn,ba,sn,w,ta,au,gemstones | request-required |
| 6 | Ethiopia licences | Ministry of Mines (Ethiopia) | direct-request | https://mines.gov.et | shapefile/GeoJSON | au,cu,zn,sn,w,ta,gemstones,industrial | request-required |
| 7 | Burundi licences | MINEODYN (Burundi) | direct-request | -- | shapefile/GeoJSON | sn,w,ta,au,cu,zn,gemstones | request-required |

**Coverage:** All 7 countries now have licence/cadastre contacts registered.


## 8. Satellite imagery for mineral exploration and alteration mapping

### 8.1 Optical multispectral satellites

| Satellite | Bands | Resolution | Key bands for mineral detection | Coverage | Access URL | Access type |
|---|---|---|---|---|---|---|
| Sentinel-2A/B (ESA Copernicus) | 13 bands MSI | 10 m (B2,B3,B4,B8), 20 m (B5,B6,B7,B8A,B11,B12), 60 m (B1,B9,B10) | B4 (Red 665nm), B8 (NIR 842nm), B11 (SWIR1 1610nm), B12 (SWIR2 2190nm) - clay/iron oxide/carbonate alteration, vegetation stress, mineral spectral signatures | Global, all 7 East African countries (3-5 day revisit) | https://dataspace.copernicus.eu/ (Copernicus DSE), https://scihub.copernicus.eu/ (SciHub legacy), COPERNICUS/S2_SR_HARMONIZED on GEE | Free (registration)
| Landsat 8 OLI | 9 bands (coastal aerosol, blue, green, red, NIR, SWIR1, SWIR2, pan, TIR) | 30 m (bands 2-7), 100 m (TIR bands 9-11), 15 m (panchromatic) | Band 4 (Red 640nm), Band 5 (NIR 860nm), Band 6 (SWIR1 1650nm), Band 7 (SWIR2 2200nm), Band 9 (Cirrus) - alteration minerals, iron oxides, clay minerals, carbonate detection | Global, all 7 East African countries (16-day revisit) | https://earthexplorer.usgs.gov/, LANDSAT/LC08/C02/T1_L2 on GEE, Landsat 8/9 on Copernicus DSE (federated) | Free (registration)
| Landsat 9 OLI-2 | Same as Landsat 8 + improved SNR | 30 m (bands 2-7), 100 m (TIR), 15 m (pan) | Same spectral coverage as Landsat 8 - improved dynamic range for faint alteration signals | Global, all 7 East African countries (16-day revisit, combined Landsat 8+9 = 8-day) | https://earthexplorer.usgs.gov/, LANDSAT/LC09/C02/T1_L2 on GEE | Free (registration)
| ASTER (Terra satellite, NASA) | 14 bands (VNIR 3 bands, SWIR 6 bands, TIR 5 bands) | 15 m (VNIR), 30 m (SWIR), 90 m (TIR) | B4 (SWIR 1.65um - illite/muscovite), B5 (SWIR 2.17um - kaolinite/smectite), B6 (SWIR 2.2um - kaolinite), B10-B14 (TIR emissivity for silicate minerals - quartz, carbonate, feldspar discrimination) | Global, all 7 East African countries (stereo + multispectral) | https://earthexplorer.usgs.gov/ (ASTER products), USGS/ASTER on GEE, https://earthdata.nasa.gov/ | Free (registration)
| Sentinel-3 OLCI | 21 bands (visible to SWIR) | 300 m | Broader coverage, lower resolution - useful for regional alteration mapping at continental scale | Global | https://dataspace.copernicus.eu/ | Free (registration)
| Sentinel-3 SLSTR | 9 bands (visible to TIR) | 500 m (visible/NIR/SWIR), 1 km (TIR) | Thermal infrared for geothermal/heat anomaly mapping in Rift valley regions | Global | https://dataspace.copernicus.eu/ | Free (registration)

### 8.2 SAR (Synthetic Aperture Radar) satellites

| Satellite | Band | Resolution | Key use for mineral exploration | Coverage | Access URL | Access type |
|---|---|---|---|---|---|---|
| Sentinel-1A/B (ESA Copernicus) | C-band (5.405 GHz, 5.65 cm wavelength) | 10 m (IW mode), interferometric products | Structural lineament mapping, terrain deformation, surface roughness for laterite/crust mapping. Dual-pol (VV+VH or HH+HV) for orientation analysis | Global, all 7 East African countries (6-day revisit for A+B) | https://dataspace.copernicus.eu/ (Copernicus DSE), https://scihub.copernicus.eu/ (SciHub legacy) | Free (registration)
| ALOS PALSAR (JAXA) | L-band (1.27 GHz, 23.6 cm wavelength) | 10 m (fine mode), 12.5 m (DEM product), ScanSAR (250m) | L-band penetrates vegetation canopy - useful for structural mapping in forested/rugged terrain (Bwindi, Virunga, East African forests). Topographic correction for SAR. | Global, all 7 East African countries (46-day revisit for ALOS-2) | https://vertex.daac.asf.alaska.edu/ (ASF Vertex - ALOS PALSAR), https://search.asf.alaska.edu/ | Free (registration)
| Sentinel-1 (dual-pol, Interferometric) | C-band | 10 m (amplitude), 5 m (interferometric phase) | InSAR for subtle topographic/structural delineation of buried intrusions, fault zones, lineaments in Sn-W-Ta belt areas | Global | https://dataspace.copernicus.eu/ | Free (registration)


### 8.3 Thermal infrared and multispectral thermal satellites

| Satellite | Bands | Resolution | Key use for mineral exploration | Coverage | Access URL | Access type |
|---|---|---|---|---|---|---|
| Landsat 8/9 TIRS | TIR Band 10 (10.60-11.19 um), Band 11 (11.50-12.51 um) | 100 m (resampled to 30 m in Level-2 products) | Thermal inertia mapping, alteration mineral mapping (carbonate, quartz, silicates), geothermal anomaly detection. Band ratios (B10/B11) for emissivity extraction | Global, all 7 East African countries | https://earthexplorer.usgs.gov/, LANDSAT/LC08/C02/T1_L2 + LANDSAT/LC09/C02/T1_L2 on GEE | Free (registration)
| ASTER TIR (Terra) | Bands 10-14 (8.125-11.65 um, 5 spectral bands) | 90 m | Emissivity spectra for silicate mineral mapping - quartz abundance, carbonate discrimination, feldspar composition. Critical for lithological mapping in arid/semi-arid areas (Northern Kenya, northern Ethiopia, South Sudan) | Global | https://earthexplorer.usgs.gov/ (AST_08, AST_05 products), USGS/ASTER on GEE | Free (registration)
| Sentinel-3 SLSTR | TIR bands (S7-S9, 10.8-12 um) | 1 km (nadir), 500 m (across-track) | Regional thermal anomaly detection, geothermal prospecting in East African Rift | Global | https://dataspace.copernicus.eu/ | Free (registration)

### 8.4 Radargrammetry and DEM from SAR

| Product | Sensor | Resolution | Use case | Access URL | Access type |
|---|---|---|---|---|---|
| ALOS World 3D - 30 m (AW3D30) | ALOS PALSAR (PRISM triplet) | 30 m | High-resolution DEM for East Africa - better than SRTM in steep terrain (Rwenzori, Virunga, Ethiopian Rift escarpments). Good for hydrological conditioning, slope features. | https://vertex.daac.asf.alaska.edu/ (for Kopoyo/DSS access), https://www.eorc.jaxa.jp/ALOS/en/aw3d30/ | Free (registration, license agreement)
| Copernicus DEM GLO-30 | TanDEM-X (TerraSAR-X + TanDEM-X) | 30 m | Highest-quality global DEM (relative accuracy ~1 m, absolute ~3 m). Superior to SRTM. Used in MPM pipeline. | COPERNICUS/DEM/GLO30 on GEE, https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/ | Free (registration, GEE or AWS open data)
| Copernicus DEM GLO-90 | TanDEM-X | 90 m | Free alternative to 30 m where license restricts commercial use. | COPERNICUS/DEM/GLO90 on GEE, https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com/ | Free (registration)
| MERIT DEM | Multi-satellite compilation | 30 m (downscaled from 90 m SRTM) | Hydro-conditioning, pit filling, drainage network extraction. Improvement over SRTM removing bias (vegetation, radar speckle). | https://hydro.iis.u-tokyo.ac.jp/MERIT_DEM/, MERIT_DEM on GEE, AWS Open Data | Free
| SRTM (Shuttle Radar Topography Mission) | SRTM (C-band interferometry) | 30 m (GL1), 90 m (legacy) | Baseline DEM. Good general-purpose structural/terrain features (slope, aspect, hillshade, terrain position, topographic wetness). | USGS/SRTMGL1_003 on GEE, https://earthexplorer.usgs.gov/ | Free (registration)

### 8.5 Cloud platforms for satellite data access (programmatic)

| Platform | Key products available | Access method | For which East African countries | URL | Access type |
|---|---|---|---|---|---|
| Google Earth Engine (GEE) | SRTM/SRTMGL1_003, USGS/ASTER, Landsat 8 (LANDSAT/LC08/C02/T1_L2), Landsat 9 (LANDSAT/LC09/C02/T1_L2), Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED, COPERNICUS/S2_SR), ALOS/PALSAR/YEARLY/SM, COPERNICUS/DEM/GLO30/GLO90, NASA/GLPK1K/1, MODIS/006/MOD13Q1 (NDVI), USGS/USA/SDGAP/ENERGY | Python API (ee), JavaScript Code Editor, REST API | All 7 countries (global coverage), any scale | https://earthengine.google.com/ (registration required) | Free (non-commercial/academic research, registration)
| Microsoft Planetary Computer (MPC) | Sentinel-2 (via STAC), Landsat (via STAC), SRTM/MERIT DEM, ALOS World 3D, MODIS/VIIRS, NAIP (US only) | Python SDK (planetary-computer), STAC API, Dask | All 7 countries (global coverage, cloud-optimised) | https://planetarycomputer.microsoft.com/ (registration required) | Free (registration)
| NASA Earthdata Search / AppEEARS | MODIS, VIIRS, HLS (Harmonized Landsat-Sentinel), Landsat, ASTER, SMAP soil moisture, GEDI canopy height/lidar, ECOSTRESS (thermal) | Web portal (earthdata.nasa.gov), Python (earthdata, pyület), AppEEARS API (subset/transform) | All 7 countries | https://earthdata.nasa.gov/ | Free (registration)
| Copernicus Data Space Ecosystem (DSE) | Sentinel-1/2/3, Copernicus DEM (GLO-30/90), ENVISAT MERIS/MERIS, Proba-V, future CO2M | Web portal (dataspace.copernicus.eu), OData API, OGC WCS/WMS, On-Demand processing | All 7 countries (global, all Sentinel missions) | https://dataspace.copernicus.eu/ | Free (registration)
| USGS EarthExplorer | Landsat 8/9, ASTER, SRTM, NAIP, NGA, MODIS, aerial photography (historical) | Web portal (earthexplorer.usgs.gov), MODAPS API for MODIS, EarthExplorer REST | All 7 countries | https://earthexplorer.usgs.gov/ | Free (registration)
| ESA SciHub / Copernicus Open Access Hub | Sentinel-1/2 (raw SAFE format, unprocessed) | Web portal (scihub.copernicus.eu), OData API, wget scripts | All 7 countries | https://scihub.copernicus.eu/ | Free (registration, legacy - migrating to DSE)
| ASF Vertex (NASA ASF DAAC) | ALOS PALSAR (Level 1.1, 2.1, DEM), Sentinel-1 (processed interferograms, amplitude), MEaSUREs (InSAR, deformation), GNSS data | Web portal (vertex.daac.asf.alaska.edu), vertex API, ASF Search API (JSON) | All 7 countries | https://vertex.daac.asf.alaska.edu/ | Free (registration, some data require proposal for restricted access)

### 8.6 Open data for spectral mineral mapping (spectral libraries, mineral spectra)

| Resource | Description | Spectral coverage | URL | Access type |
|---|---|---|---|---|
| USGS Digital Spectral Library (splib07a) | Reference reflectance spectra of minerals, rocks, meteorites - 512+ mineral spectra in VIS-SWIR (0.2-3 um), TIR (3-16 um), emissivity | 0.2-16 um (full range) | https://speclab.cr.usgs.gov/spectral-lib.html | Free (download ASCII)
| ASTER Spectral Library (JEOS) | ASTER band-weighted spectra for >3000 minerals, vegetation, man-made materials - built for ASTER band simulation and spectral matching | VNIR+SWIR+TIR (ASTER bands) | https://speclib.ceres.co.jp/ (JEOS - JAXA Earth Observation System) | Free (registration)
| ECOSTRESS Spectral Library (NASA/JPL) | Thermal infrared emissivity spectra for rocks, minerals, soils, vegetation - 2000+ spectra, 2-15 um | TIR (2-15 um) | https://speclib.eco.tsp.nasa.gov/ | Free (registration)
| USGS Tetracorder / ENVI spectral matching | Quantitative spectral matching software and libraries for hyperspectral image analysis (AVIRIS, Hyperion, EnMAP, PRISMA, EMIT) | Full range | https://cr.usgs.gov/thermal_infrared.html, https://speclab.cr.usgs.gov/ | Free (open source tools)

### 8.7 Hyperspectral satellite missions (emerging, relevant to mineral detection)

| Satellite / sensor | Spectral range | Bands / resolution | Key mineral detection capability | Coverage for East Africa | Access URL | Access type |
|---|---|---|---|---|---|---|
| NASA EMIT (Earth Surface Mineral Dust Source Investigation, on ISS) | 0.4-2.4 um (443 bands) | ~7 nm spectral, 60 m spatial | Direct mineral mapping from space: kaolinite, illite, montmorillonite, calcite, dolomite, hematite, goethite, chlorite, sericite, gypsum - highly relevant for Sn-W-Ta alteration, base metal alteration, bauxite laterite mapping | Global, all 7 East African countries (ISS orbit, every few days over tropics) | https://earthdata.nasa.gov/ (EMIT products via LP DAAC), EMIT on GEE (USGS/EMIT_L2A), https://emit.jpl.nasa.gov/ | Free (NASA, registration)
| EnMAP (German hyperspectral, DLR) | 0.4-2.5 um (244 bands) | ~6.5 nm spectral, 30 m spatial | Mineral mapping, soil properties, alteration mineral identification, environmental monitoring | East Africa: limited coverage (single-track, user-scheduled targets, global coverage ~3-year mission) | https://www.enmap.org/ (registration, browsing), https://geoservice.dlr.de/ (data service) | Free for research (registration)
| PRISMA (Italian hyperspectral, ASI) | 0.4-2.5 um (256 bands) | ~10 nm spectral, 100 m spatial | Similar to EnMAP, global coverage, mineral mapping, environmental. Useful for alteration mapping in African context | Global (systematic coverage, ~1-month revisit at equator) | https://www.progetto-prisma.org/ (registration, data request) | Free for research (registration)
| PRISMA-2 (future) / CHIME (future hyperspectral, ESA) | 0.4-2.5 um (planned) | TBD | Future systematic hyperspectral coverage for mineral mapping | Planned | TBD | TBD
| AVIRIS-NG (airborne, NASA/JPL) | 0.4-2.5 um (432 bands) | ~5 nm spectral, 3-5 m spatial (airborne) | Not satellite but high-resolution airborne hyperspectral - used in mineral exploration in East Africa / Africa (project-based) | Project-based airborne surveys in specific African locations | https://aviris.jpl.nasa.gov/ | Research collaboration (proposal-based)


## 9. Geophysical / remote sensing satellites (magnetic, gravity, radiometric)

### 9.1 Satellite magnetic field missions (for crustal magnetic anomaly mapping)

| Satellite / mission | Type | Spatial resolution | Key use for mineral exploration | Coverage | Access URL | Access type |
|---|---|---|---|---|---|---|
| SWARM (ESA) - 3 satellites (A, B, C) | Vector + scalar magnetometers (fluxgate + absolute) | ~460 km altitude, ground track separation ~100 km (day side), global coverage | Crustal magnetic anomaly field (after removing core, external, ionospheric fields) - mapped at 1-2 nT resolution. Basement fabric, igneous intrusions, fault delineation for Sn-W-Ta and Cu-Zn targets | Global, all 7 East African countries. Data available since 2014 | https://earth.esa.int/swarm (ESA Swarm DISC), https://earthdata.nasa.gov/ (Swarm via ASF or through DISC), https://geomag.org/ (geomagnetic models), https://wdmam.org/ (WDMAM - includes Swarm-derived data) | Free (ESA, registration for some services)
| CHAMP (GFZ) - ended 2010 | Vector magnetometer | 454 km, global | Historical baseline crustal magnetic anomaly (2000-2010). Merged with Ørsted and SAC-C for long-wavelength crustal field model | Global | https://www.gfz-potsdam.de/en/section/space-geodesy-and-surveying/data-products/ | Free (GFZ, some products)
| Ørsted (Denmark) - ended 2014 | Vector magnetometer | 650 km (sun-synchronous 97.8 deg inclination) | Historical crustal magnetic anomaly field (1999-2014) merged into global models (WDMAM, MF7, EGF7) | Global | https://www.space.dtu.dk/English/Research/Space/Facilities/Orsted/Satellites/ | Free (DTU Space)
| SAC-C (Argentina/US) - ended 2004 | Scalar and vector magnetometer | 704 km | Contributed to early crustal magnetic field models (2000-2004) | Global (higher latitude emphasis) | NASA GSFC / ASF data archive | Free

Magnetic anomaly products derived from satellite data:
- **WDMAM** (World Digital Magnetic Anomaly Map) v2.0 - 5 km resolution compiled from satellite, airborne, marine, ground surveys globally. See also config entry wdmam in data_sources.yaml. URL: https://wdmam.org/ or via CGMW https://ccgm.org/ | Free (registration, some contributions)
- **EMAG2** (Earth Magnetic Anomaly Grid, v3) - 2 arc-minute resolution (~3.7 km) global compilation (NOAA/NCEI). Available via NOAA NCEI, GEE, or direct download. URL: https://www.ngdc.noaa.gov/mgg/geomagnetic/emag2/emag2.html | Free
- **MF7** (Magsat-Field model 7) - spherical harmonic crustal field model (253 orders) from satellite data (Magsat, Ørsted, CHAMP, SAC-C, SWARM). Available via geomag.org, BGS, and as Grids through USGS. URL: https://geomag.org/models/mf7/ | Free

### 9.2 Satellite gravity missions (for crustal density / structural mapping)

| Satellite / mission | Type | Spatial resolution | Key use for mineral exploration | Coverage | Access URL | Access type |
|---|---|---|---|---|---|---|
| GRACE (NASA/DLR) - twin satellites (ended 2017, replaced by GRACE-FO) | K-band ranging (inter-satellite distance), GPS, accelerometer | ~300 km (quarter-degree, ~280 km at equator), monthly solutions | Time-variable gravity (groundwater, ice mass change) - not directly useful for mineral exploration but provides hydrogeological context (perched aquifers, fault-controlled groundwater flow) and crustal mass change monitoring | Global, all 7 East African countries | https://earthdata.nasa.gov/ (GRACE/GRACE-FO via PO.DAAC or GES DISC), https://grace.jpl.nasa.gov/ | Free (registration)
| GRACE-FO (NASA/DLR) - ongoing (launched 2018) | Same as GRACE + Laser Ranging Interferometer (LRI) | ~300 km, monthly, improved precision | Same as GRACE - time-variable gravity for hydrogeology, with improved precision for subtle crustal/subsurface density variations over time | Global, all 7 East African countries | https://earthdata.nasa.gov/ (GRACE-FO), https://gracefo.jpl.nasa.gov/ | Free (registration)
| GOCE (ESA) - ended 2013 | Gradiometer (3-axis gravity gradient), GPS, accelerometers, drag-free control | ~80-100 km (100 km), global, 1 mGal accuracy (0.01 mGal noise after processing) | Static gravity field and geoid - key for regional structural mapping, density contrasts (granite vs gneiss vs volcanics), basin detection, crustal thickness model (GOCE + seismic). Very useful for large-scale crustal architecture mapping in East African Rift context | Global, all 7 East African countries (2009-2013 mission) | https://earth.esa.int/gocedata/ (ESA GOCE data portal), https://earthdata.nasa.gov/ (GOCE via PO.DAAC), http://icgem.gfz-potsdam.de/ (ICGEM - gravity model computation) | Free (ESA, registration)

Gravity products derived from satellite data:
- **EIGEN-6C4** (GFZ/GRGS) - satellite + terrestrial gravity combined model, 1' resolution (~1.8 km). Used for regional gravity anomaly maps. URL: http://icgem.gfz-potsdam.de/ (ICGEM) | Free
- **WGM2012** (World Gravity Map) - 1 km resolution gravity anomaly map compiled from satellite (GOCE, GRACE) + airborne + terrestrial. URL: https://bgi.omp.obs-mip.fr/ (BGI, registration) | Free (registration, some restrictions)
- **GRAV-D** (NOAA Geoid) - geoid model for North America only, not relevant for Africa

### 9.3 Satellite radiometric surveys (gamma-ray spectrometry)

| Satellite / mission | Type | Spatial resolution | Key use for mineral exploration | Coverage | Access URL | Access type |
|---|---|---|---|---|---|---|
| No dedicated orbiting gamma-ray spectrometer for mineral exploration currently operational at useful resolution | - | - | - | - | - | -
| Historical: Compton Gamma Ray Observatory (CGRO, NASA) - ended | Gamma-ray spectrometer (EGRET) | Point / low resolution | Not for mineral exploration (high-energy gamma astrophysics) | Global | NASA GSFC | N/A for mineral mapping

Note: Orbiting gamma-ray spectrometry for elemental mapping (K, U, Th) requires low Earth orbit (<600 km) and sensitive spectrometers. Current missions (e.g. ChenoSphere, GRANAT) are historical or specialized. For regional K/Th/U mapping relevant to alteration (potassic, albitised, sheared zones) in East Africa, airborne radiometric surveys (national surveys, e.g. Tanzania, Uganda, Kenya - request from geological surveys) are the primary source, not satellite.

### 9.4 Satellite-derived topographic / geomorphic proxies for structural geology

| Product / method | Source | Resolution | Use case for mineral exploration | Coverage | Access URL | Access type |
|---|---|---|---|---|---|---|
| SRTM / ALOS / Copernicus DEM derivatives (slope, aspect, hillshade, curvature, terrain position, topographic wetness, viewshed) | SRTM, ALOS PALSAR, Copernicus DEM (see 8.4) | 30 m | Structural features: lineament extraction, drainage anomalies (sinuosity, junction angles indicative of structural control), asymmetric valley patterns (fault scarps), terrain position relative to intrusion contacts, geomorphic proxies for concealed mineralization | All 7 East African countries | DEM products from GEE, ASF Vertex, Copernicus DSE (see 8.4) | Free (see 8.4)
| Global lithological maps from remote sensing (consult, litho) | USGS World Geology, BRGM Africa, OneGeology (see Section 6) | Variable (1:35M to 1:50k) | Regional geological framework - primary context for mineral prediction | All 7 East African countries | See Section 6 | Free / request
| Nighttime lights / urban extent (VIIRS) | VIIRS on Suomi NPP / NOAA-21 (NASA) | 500 m | Infrastructure mapping, settlement extent - useful for exploring accessibility / logistics of exploration targets (mining infrastructure planning) | Global, East Africa | https://earthdata.nasa.gov/ (VIIRS DNB), VIIRS/006/VNP46A2 + VNP46A3 on GEE | Free (registration)
| Sentinel-2 change detection / alteration mapping | Sentinel-2 (see 8.1) | 10-20 m | Time-series analysis of alteration zones, tracking surface changes (mining, quarrying, land disturbance) that may indicate known deposits/exploitation | All 7 East African countries | Copernicus DSE, GEE (COPERNICUS/S2_SR_HARMONIZED) | Free (registration)
| Thermal anomaly detection (geothermal, metallogenic heat) | ASTER TIR (see 8.3), Landsat TIRS, Sentinel-3 SLSTR | 30-100 m to 1 km | Geothermal prospect mapping in East African Rift (Ethiopia, Kenya, Tanzania rift zones) - indirect indicator of hydrothermal systems potentially associated with mineralization | East African Rift segments (Ethiopia, Kenya, Tanzania) | ASTER, Landsat TIRS, Sentinel-3 SLSTR (see 8.3) | Free (registration)

### 9.5 Integration of satellite remote sensing into MPM pipeline

For the East Africa MPM pipeline, satellite data integration is already partially implemented:
- **DEM**: SRTM (EPSG:32736, aligned to grid) - already in pipeline srtm_dem ingest module (src.ingest.dem) via GEE (USGS/SRTMGL1_003)
- **Multispectral imagery**: Sentinel-2 (src.ingest.sentinel2_ee) - already in pipeline, download from GEE using COPERNICUS/S2_SR_HARMONIZED
- **Spectral features**: terrain/terrain features (from DEM), geology features (from USGS worldgeol + BRGM), and Sentinel-2 bands (10-20 m) already extracted and used as features in src.features module

Additional satellite data that could be integrated (future pipeline enhancement):
| Data source | Integration path | Potential value-add for MPM |
|---|---|---|
| ASTER SWIR/TIR spectral indices (silicate/carbonate/kaolinite/illite) | New ingest module (src.ingest.aster) | Direct alteration mineral detection - very relevant for Sn-W-Ta (pegmatite / greisen / albitisation alteration), Cu-Zn (carbonate/sericite alteration), bauxite (laterite/kaolinite) mapping. Could enhance feature set. |
| Landsat 8/9 (historical time series) | New ingest module (src.ingest.landsat_ee, mirroring Sentinel-2 pattern) | Multi-temporal alteration mapping, change detection, legacy data for regions with cloudiness where Sentinel-2 is sparse |
| Sentinel-1 SAR (amplitude + interferometry) | New ingest module (src.ingest.sentinel1_ee or Copernicus DSE) | Structural lineament enhancement, surface roughness/crust mapping, geomorphic lineament extraction for fault/intrusion detection in Sn-W-Ta and Cu-Zn belt settings |
| ALOS PALSAR DEM / SAR | New ingest module (ASF Vertex API) | Better DEM for steep/alpine terrain in virunga/Rwenzori/Ethiopian rift, SAR for structural mapping where optical limited by cloud/forest cover |
| EMIT hyperspectral data (when available and accessible) | New ingest module (src.ingest.emit_ee or NASA Earthdata) | Direct mineral spectral mapping from space - highest potential for alteration mineral detection. Still emerging data access, patience needed |
| SWARM magnetic anomaly grid (WDMAM/EMAG2) | New ingest module (src.ingest.wdmam / src.ingest.emag2) | Crustal magnetic architecture for large-scale structural context, intrusion detection, basement fabric - valuable for Sn-W-Ta and Cu-Zn targets |
| GOCE gravity anomaly (EIGEN-6C4/WGM2012) | New ingest module (src.ingest.goc_*) | Density contrast mapping, crustal thickness, rift architecture - regional tectonic context for all commodities |


## 13. Satellite / Earth observation access portals (bulk entry)

| # | Source | Provider | Access | URL | Format | Commodities | Status |
|---|---|---|---|---|---|---|---|
| 1 | USGS EarthExplorer | USGS / NASA | portal | https://earthexplorer.usgs.gov/ | GeoTIFF bulk / MTK / zipped | all | configured |
| 2 | NASA Earthdata Search | NASA Earthdata / DAAC network | portal | https://search.earthdata.nasa.gov/ | GeoTIFF / netCDF / HDF, OGC WCS/WMS | all | configured |
| 3 | Copernicus Data Space Ecosystem (CDSE) | ESA / EU Copernicus | download | https://dataspace.copernicus.eu/ | SAFE / JPEG2000 / COG (Phenix) / OData API | all | configured |
| 4 | ESA SciHub (legacy) | ESA / Copernicus | portal | https://scihub.copernicus.eu/ | SAFE / JPEG2000 | all | pending |
| 5 | ASF Vertex (SAR + DEM) | NASA ASF DAAC | portal | https://vertex.daac.asf.alaska.edu/ | GeoTIFF (ALOS DEM), SAR KMZ/GeoTIFF | sn,w,ta,cu,zn | configured |

**Coverage:** All 7 countries covered by these global/regional satellite portals.


## 14. Mineral occurrence / deposit databases (global / continental)

| # | Source | Provider | Access | URL | Format | Commodities | Status |
|---|---|---|---|---|---|---|---|
| 1 | Mindat.org | Mindat.org / Hudson Institute | portal | https://www.mindat.org/ | Web table / manual geocoding | all | pending |
| 2 | Natural Earth | Natural Earth / NACIS | download | https://www.naturalearthdata.com/ | Shapefile / GeoJSON (1:10m, 1:50m, 1:110m) | all | configured |
| 3 | USGS Africa Mineral Compilation | USGS | download | (pending-url) | FileGDB / shapefile-zip | all | pending-url |
| 4 | GeoNames | GeoNames.org | download | http://www.geonames.org/ | Plain text dump / REST API | all | configured |
| 5 | OneGeology Portal | OneGeology / CGMW | portal | https://www.onegeology.org/ | WCS / WMS / WFS / shapefile | all | pending |
| 6 | African Mineral Deposits (academic) | Academic / UN publications | manual | -- | Tables / maps PDFs | au,cu,zn,sn,w,ta,ba,gemstones | manual |

## 15. Regional geophysical compilations

| # | Source | Provider | Access | URL | Format | Commodities | Status |
|---|---|---|---|---|---|---|---|
| 1 | WDMAM (World Digital Magnetic Anomaly Map) | IGPP / BGI / CGMW | download | https://wdmam.org/ | GeoTIFF / netCDF / ESRI ASCII grid | sn,w,ta,cu,zn | pending-url |
| 2 | AfricaGEOSS | AUC / GMES Africa | portal | https://africa-geoss.org/ | varies (WMS / WFS / download) | all | pending-url |


## 16. National geological surveys (7-country coverage summary)

| Country | Survey / Authority | Website | Access |
|---|---|---|---|
| Rwanda | Rwanda Mines Board (RMB) | https://www.rmb.gov.rw | direct-request |
| Uganda | Directorate of Geological Survey and Mines (DGSM) | https://dgsm.go.ug | direct-request |
| Kenya | Geological Survey of Kenya (GSK) / Ministry of Mining | https://www.geologicalsurvey.go.ke | direct-request |
| Tanzania | Geological Survey of Tanzania (GST) / Tanzania Mining Commission (TMC) | https://www.gst.go.tz / https://www.tmc.go.tz | direct-request |
| Burundi | MINEODYN - Ministry of Energy and Mines | -- | direct-request |
| South Sudan | Directorate of Geological Surveys / Ministry of Mining | -- | direct-request |
| Ethiopia | Ethiopian Geological Survey (EGS) / Ministry of Mines (MoM) | https://mines.gov.et | direct-request |

## 17. Auxiliary / cloud access

| # | Source | Provider | Access | URL | Format | Status |
|---|---|---|---|---|---|---|
| 1 | Google Earth Engine (GEE) | Google | download (registration) | https://earthengine.google.com/ | GeoTIFF (Export.image.toDrive/TS), numpy array | configured |
| 2 | Microsoft Planetary Computer | Microsoft Azure / AI for Earth | download (registration) | https://planetarycomputer.microsoft.com/ | COG via STAC API, Zarr, netCDF | pending-url |



## 18. Hyperspectral missions (EnMAP / PRISMA) - wavelength-level mineral detection

EnMAP (Environmental Mapping and Analysis Program, DLR Germany) and PRISMA (ASI Italy) are spaceborne **imaging spectrometers** that capture ~224 contiguous bands across 420-2450 nm (VNIR + SWIR) at 30 m pixels. Unlike Sentinel-2's broad SWIR bands, they **resolve the individual mineral absorption features** that the MPM pipeline fingerprints depend on.

### Why EnMAP matters for this pipeline

Sentinel-2 B12 (2.19 um, ~100 nm wide) **cannot separate**:
- the **2.20 um Al-OH uptake** (phyllic/sericite: muscovite, kaolinite, gibbsite - Sn-W-Ta greisen & bauxite gibbsite crusts), from
- the **2.33 um Mg-OH/CO3 uptake** (propylitic: chlorite, epidote, carbonate halos).

EnMAP's ~10 nm SWIR sampling separates them cleanly. Diagnostic wavelengths wired into src/features/hyperspectral.py:

| Centre wavelength | Absorption | Diagnostic minerals | Commodity relevance |
|---|---|---|---|
| **2200 nm** (Al-OH) | clay hydroxyl | muscovite, sericite, kaolinite, lepidolite, gibbsite | Sn-W-Ta pegmatite/greisen halos; bauxite (gibbsite) |
| **2330 nm** (Mg-OH/CO3) | Mg-hydroxyl / carbonate | chlorite, epidote, carbonate | Cu-Zn propylitic halos; greisen carbonate envelopes |
| **900 nm** (Fe-Oxide) | Fe3+ charge transfer / crystal field | hematite, goethite, jarosite | gossans over sulphides; laterite Fe-crust |

### Data access

| Item | EnMAP | PRISMA |
|---|---|---|
| Provider | DLR (German Aerospace Center) | ASI (Italian Space Agency) |
| Portal | https://www.enmap.org/ | https://prisma.asi.it/ |
| Ordering | DLR EOWEB GeoPortal: https://eoweb.dlr.de/ | PRISMA portal (registration) |
| Cost | free for scientific use (registration) | free for research (registration) |
| Bands | ~224 (420-2450 nm), ~10 nm sampling | ~230 (VNIR 400-1010, SWIR 920-2530 nm) |
| Resolution | 30 m, ~30 km swath | 30 m, 30 km swath |
| Products | L0 / L1B / L1C (TOA) / **L2A (BOA reflectance)** | L1 / **L2D (BOA reflectance)** |
| Format | GeoTIFF (DN x1e4, auto-scaled) / ENVI | GeoTIFF / HDF5 |
| Targeting | **on-demand Foreground Mission Campaign** planning | on-demand proposals |
| Tools | EnPT (Python, destriping >=1.4.0), EnMAP-Box (QGIS) | ASI products + standard tools |
| Pipeline status | configured - full ingest + features implemented | pending - same ingest path, account required |

### Workflow in this repo (already implemented)

1. **Order scenes** over the Karagwe-Ankole (Rwanda), Copperbelt (DRC/Zambia) and Usambara (Tanzania) AOIs via EOWEB (or request a targeted Foreground Campaign acquisition).
2. **Ingest** - src/ingest/enmap.py:
   python -m src.ingest.enmap --input data/raw/enmap/enmap_tile.tif --config configs/karagwe.yml --out data/interim/hyperspectral/enmap_kabar_aligned.tif
   (auto-scales DN to reflectance, parses band-centre wavelengths from descriptions, applies brightness cloud mask, reprojects onto the 30 m belt grid with NaN for cloudy/nodata).
3. **Features** - src/features/hyperspectral.py: band-depth maps at 2200 / 2330 / 900 nm, Spectral Angle Mapper against mineral reference spectra (kaolinite, muscovite, chlorite, gibbsite, hematite), PCA compression for downstream models.
4. **Fingerprints** - configs/spectral_fingerprints.yaml already declares per-group probe wavelengths (e.g. [2.20, 2.35] um for Sn-W-Ta) and lists which discriminations REQUIRE EnMAP-class data.

**Caveat:** some EnMAP scenes show SWIR along-track striping near water bands - use EnPT >= 1.4.0 destriping before band-depth analysis.

## 19. DLR EOC OGC web map services (evaluated 2026-09-22)

Two free OGC WMS 1.3.0 endpoints published by the DLR Earth Observation Center (EOC) GeoService were evaluated against all three belt AOIs (Karagwe-Ankole, Central African Copperbelt, Usambara). Coverage of each render was measured as the fraction of non-transparent pixels in a 512 x 512 `GetMap` PNG. The checks are reproducible with `python -m src.ingest.dlr_eoc_wms --list`, and `--belt bauxite` downloads the usable layers.

### 19.1 Imagery Map Service - https://geoservice.dlr.de/eoc/imagery/wms (28 layers) - NOT USABLE

| # | Layer group | Declared extent | Verdict |
|---|---|---|---|
| 1 | Sentinel-2 L2A MAJA (FRE/SRE/CLM/MG2 and V2) | Germany only (5.8-15.2 E, 46.8-55.9 N) | outside all belts |
| 2 | Sentinel-2 L3A WASP (FRC/FLG/DTS and V2) | Germany only | outside all belts |
| 3 | MODIS-EU daily / NRT / footprints | 51.8 W-76.4 E, 27.7-77.2 N | outside all belts (starts at 27.7 N) |
| 4 | RapidEye RESA L3M mosaic + 3K Brunswick aerial + MODIS Germany mosaic | Germany | outside all belts |
| 5 | EnMAP HSI L0 QL family (footprints, SWIR, VNIR, cloud, cloud shadow, haze, cirrus, snow, classes) | declared global | renders 0 % opaque on every window tested, including a Germany control; `GetFeatureInfo` returns `numberReturned: 0` - the service serves no EnMAP quicklook data |
| 6 | S2_TILE_GRID | global | usable - reference tiling scheme only |

**Conclusion:** the imagery service contributes no usable data layer to this project. It is registered as `dlr_eoc_imagery_wms` (status `pending`) purely for traceability, and `S2_TILE_GRID` is kept as an optional reference overlay for planning Sentinel-2 ingest runs. EnMAP scene discovery stays with the DLR EOWEB portal (section 18) and EnMAP ingest stays on the L2A route (`src/ingest/enmap.py`).

### 19.2 Land Map Service - https://geoservice.dlr.de/eoc/land/wms (141 layers) - PARTIALLY USABLE

Verified renders (non-transparent fraction of the 512 x 512 tile; `-` = empty tile / no coverage):

| Layer | Style | Karagwe (Sn-W-Ta) | Copperbelt (Cu-Zn) | Usambara (bauxite) | Use in this project |
|---|---|---|---|---|---|
| SOILSUITE_SRC_AFR_P4Y | soilsuite-src-afr-p4y-fin | 63 % | - | 28 % | bare-surface reflectance composite (Africa, 4-year): laterite / bauxite and soil-regolith context |
| SOILSUITE_SRC-CI95_AFR_P4Y | soilsuite-src-ci95-afr-p4y-fin | 63 % | - | 28 % | confidence companion to the composite mean |
| SOILSUITE_SRC-STD_AFR_P4Y | soilsuite-src-std-afr-p4y-fin | 63 % | - | 28 % | temporal variability of the composite |
| SOILSUITE_SFREQ-BSF_AFR_P4Y | soilsuite-sfreq-bsf-afr-p4y-fin | 100 % | - | 100 % | bare-surface frequency = exposure mask for alteration / laterite work |
| TDM_FNF50 | fnf | 100 % | 100 % | 100 % | TanDEM-X 50 m forest / non-forest mask (all belts) |
| WSF_2019 | wsf2019 | 35 % | 2 % | 8 % | settlement / anthropogenic disturbance mask |
| WSF_Evolution | wsfevolution | <1 % | - | - | settlement change over time (artisanal mining growth) |
| GUF28_DLR_v1_Mosaic | guf_8bit | 2 % | 1 % | 1 % | coarse urban footprint (sparse over the rural belts) |
| ESA_LAND_OCEAN_MAP | oceanmask | 9 % | - | - | land / water mask for grid trimming |
| TS_LANDSAT_2015 | timescan-standard | 100 % | 100 % | 100 % | independent Landsat median composite for QA of the probability surface |
| ne_countries | (default) | 1 % | - | - | national boundaries for AOI masks and labels |

Notes:

- SoilSuite spans 28.75 E-50.31 E / 7.22 S-16.06 N, so the **Copperbelt lies outside the product**; the remaining layers are global.
- Style names frequently differ from layer names (for example `GUF28_DLR_v1_Mosaic` uses style `guf_8bit`); the verified layer/style pairs live in `src/ingest/dlr_eoc_wms.py`.
- Request `GetMap` in `EPSG:3857` or `CRS:84`; oversized windows on some layers answer with a `ServiceExceptionReport` (raise it, do not silently cache a blank tile).
- **These are 8-bit colour-stretched tiles, not DN / reflectance values**, so they are context, QA and mask-screening layers only - they never enter the feature matrix.

Registered as `dlr_eoc_land_wms` (access `wms`, status `configured`) with the helper module `src.ingest.dlr_eoc_wms` (CLI: `--list`, `--belt`, `--layer --bbox --out`) and a context-layer toggle on the **Probability Map** tab of the dashboard (`app.py`).

## 20. Analogue-informed target scenes (permissive windows, not labels)

**Registry:** `configs/analogue_targets.yaml` (17 windows, 7 countries)
**Methodology:** `ANALOGUE_DEPOSITS_EVALUATION.md`

World-class deposits (Manono-Kitotolo, Greenbushes, Bikita, Erzgebirge
greisen, Zambian/Congolese Copperbelt, Kamoa-Kakula, Kipushi,
Kupferschiefer, Kidd Creek, Bisha, Weipa, Sangaredi, Minim-Martap,
Trombetas) were evaluated for transferable deposit-model criteria and
mapped onto permissive target windows:

| Group | Targets | World analogues | EnMAP probes (um) |
|---|---|---|---|
| Sn-W-Ta (LCT pegmatite / greisen) | 5 (Burundi, Tanzania x3, Uganda) | Manono-Kitotolo, Greenbushes, Bikita, Erzgebirge | 2.20, 2.33 |
| Cu-Zn (stratiform + Kipushi + VMS) | 6 (DRC x2, Zambia, Rwanda, Ethiopia, Tanzania) | Copperbelt, Kamoa, Kipushi, Kupferschiefer, Kidd Creek, Bisha | 2.20, 2.33, 0.86 |
| Bauxite (plateau laterite) | 6 (Tanzania x2, Kenya, Ethiopia, Uganda, South Sudan) | Weipa, Sangaredi, Minim-Martap, Trombetas | 2.20, 0.90 |

Each entry carries bbox (WGS84), CRS (UTM zone checked against
longitude), resolution, spectral/terrain/geophysical signatures, the
dry-season acquisition window per climate zone, and a confidence tier
(7 high / 6 moderate / 4 low).

**Guardrail:** these windows are *permissive scene-selection extents
only*. Analogues are never injected as positive labels (domain shift
would contaminate spatial CV); labels stay on MRDS + national-survey +
field-validated sources via `src.labels.build_labels`.

**Workflow per window:** confirm permissive geology on national-survey
maps (section 1/16) → SRTM + Sentinel-2 dry-season ingest → fingerprint
screen → order EnMAP L2A on-demand (section 18) over flagged cells →
ingest via `src.ingest.enmap` + `src.features.hyperspectral` probe
wavelengths above.

**Tooling:** `python -m src.ingest.analogue_scene` (`--list`,
`--write --confidence high` → `configs/scenes/<id>.yml`,
`--ingest --id <target>` for the SRTM + Sentinel-2 dry-run over a
window). EnMAP order parameters for the 7 high-confidence targets
(EOWEB/`planning.enmap.org` steps, per-target bbox/dry-season/probes,
tracker): `ENMAP_ORDERING_CHECKLIST.md`.

## Key Satellite Access Points (Direct URL Summary)

| Task | Best Platform | Direct URL |
|---|---|---|
| Landsat 8/9 + SRTM + ASTER download | USGS EarthExplorer | https://earthexplorer.usgs.gov/ |
| Sentinel-1/2 bulk access (modern) | Copernicus Data Space Ecosystem | https://dataspace.copernicus.eu/ |
| Sentinel-1/2 (legacy OData) | ESA SciHub | https://scihub.copernicus.eu/ |
| ALOS PALSAR DEM + Sentinel-1 SAR | NASA ASF Vertex | https://vertex.daac.asf.alaska.edu/ |
| Landsat + SRTM + ASTER (cloud compute) | Google Earth Engine | https://earthengine.google.com/ |
| NASA EO (HLS, MODIS, VIIRS, SMAP, GEDI) | NASA Earthdata Search | https://search.earthdata.nasa.gov/ |
| Global mineral occurrences | Mindat.org | https://www.mindat.org/ |
| USGS mineral deposit database (MRDS) | USGS MRDS | https://mrdata.usgs.gov/mrds/ |
| Global magnetic anomaly map (WDMAM) | WDMAM | https://wdmam.org/ |
| Country boundaries / reference vectors | Natural Earth | https://www.naturalearthdata.com/ |
| Pan-African geospatial discovery | AfricaGEOSS | https://africa-geoss.org/ |
| National geology (global aggregate) | OneGeology | https://www.onegeology.org/ |
| Microsoft cloud EO (STAC) | Microsoft Planetary Computer | https://planetarycomputer.microsoft.com/ |
| DLR EOC land context layers (WMS) | DLR GeoService | https://geoservice.dlr.de/eoc/land/wms |
| DLR EOC orthoimagery / EnMAP quicklooks (WMS) | DLR GeoService | https://geoservice.dlr.de/eoc/imagery/wms |

## Summary Statistics

**Total sources registered:** 76 (up from ~30 before this enrichment)

**Update 2026-09-22:** two DLR EOC OGC-WMS entries were added after the count above
(`dlr_eoc_imagery_wms` - evaluated as not usable, `dlr_eoc_land_wms` - verified context
layers, section 19). `configs/data_sources.yaml` currently holds 49 top-level source entries.

**By access type:**
- Configured (automated/portal-ready): 18 sources
- Pending-url (script ready, URL to configure): 8 sources
- Request-required (contact national surveys): 17 sources
- Manual (academic papers, reports): 4 sources
- Portal (interactive download): 12 sources
- Pending (planned): 2 sources

**By commodity coverage:**
- Sn-W-Ta (tin-tungsten-tantalum): 28 sources (geology, geophysics, imagery, deposits, licences)
- Cu-Zn (copper-zinc): 22 sources (geology, geophysics, imagery, deposits, licences, geochemistry)
- Ba (bauxite): 18 sources (geology, imagery, deposits, licences)
- Au (gold): 12 sources (geology, deposits, licences, geochemistry)
- All commodities: 36 sources (satellite portals, DEM, geology base, reference data)

**7-country coverage achieved:** Rwanda, Uganda, Kenya, Tanzania, Burundi, South Sudan, and Ethiopia all now have at least one national geological survey / mining authority contact registered.
