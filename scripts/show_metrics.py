import sys

sys.path.insert(0, r"D:\Users\HP\COC AI\east-africa-mpm")
from src.features.priors import apply_prior_to_raster, lift_report

CASES = {
    "tin_tungsten_tantalum": r"D:\Users\HP\COC AI\east-africa-mpm"
                             r"\outputs\models\proba_tin_tungsten_tantalum_lgbm.tif",
    "copper_zinc": r"D:\Users\HP\COC AI\east-africa-mpm"
                   r"\outputs\models\proba_copper_zinc_xgb.tif",
    "bauxite": r"D:\Users\HP\COC AI\east-africa-mpm"
               r"\outputs\models\proba_bauxite_lgbm.tif",
}

for group, proba in CASES.items():
    print("=" * 60)
    print(group)
    prior_path, diag = apply_prior_to_raster(proba, group)
    print("  sources  :", diag["sources"])
    print("  shift    : mean |dP| =", round(diag["mean_abs_shift"], 4))
    lift = lift_report(proba, prior_path, group)
    print("  AP base  :", round(lift["ap_base"], 3))
    print("  AP prior :", round(lift["ap_prior"], 3))
    print("  lift     : {:+.3f}".format(lift["lift"]))
    print("  out      :", prior_path)
