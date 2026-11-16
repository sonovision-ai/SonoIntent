# -*- coding: utf-8 -*-
"""Reference implementation of the SonoIntent reasoning renderer.

Given a source-dataset record, the renderer produces the structured reference
reasoning released with SonoIntent. It selects at most two image-derived
evidence clauses per target from a target-specific profile and falls back to
frozen per-image evidence when an axis is not computable.

The renderer reads the frozen intermediate artifacts of the V1 build (the V15
sample manifest and the V17 evidence table); set SONOINTENT_SOURCE_ROOT to the
root of that frozen source tree.
"""
import collections, csv, json, os, re

import numpy as np
from PIL import Image

from . import v19_metrics as M
from . import v20_specs as SP
from .v19_profiles import PROFILES as _PROFILES

ROOT = os.environ.get("SONOINTENT_SOURCE_ROOT", "")

PROFILES = dict(_PROFILES)
EN = {
 "乳腺病灶": ("Is there an abnormal region in this image that requires further evaluation? If so, please delineate it.",
  "Breast evaluation first requires identifying a focal lesion distinct from normal breast tissue and defining its extent for subsequent sign analysis. {EVID} Therefore, delineate the lesion itself. The boundary follows the interface between the lesion and surrounding breast tissue, excluding adjacent normal tissue and perilesional acoustic shadowing."),
 "消化道壁占位": ("Is there an abnormality of the gastrointestinal wall that requires further evaluation? If so, please delineate it.",
  "Assessment of a gastrointestinal wall abnormality first requires confirming whether a focal lesion originates from or attaches to the bowel wall, and then defining its extent. {EVID} Therefore, delineate the abnormal region. The boundary follows the interface between the lesion and the normal wall layers and adjacent tissue, excluding normal bowel wall, the lumen, and surrounding tissue."),
 "肝内占位": ("Is there an abnormal region in the liver that requires further evaluation? If so, please delineate it.",
  "Liver evaluation first requires finding a focal lesion within the parenchyma and defining its extent for subsequent characterization. {EVID} Therefore, delineate the lesion itself. The boundary follows the interface between the lesion and the surrounding liver parenchyma, excluding normal parenchyma, intrahepatic vessels, and adjacent organs."),
 "卵巢肿块": ("Is there an adnexal abnormality that requires further evaluation? If so, please delineate it.",
  "Adnexal evaluation first requires determining whether a focal mass related to the ovary or adnexa exists, and then defining the extent of the mass. {EVID} Therefore, delineate the mass itself. The boundary follows the outer margin of the mass, without including the whole adnexal region, normal ovarian tissue, or surrounding pelvic structures."),
 "甲状腺结节": ("Is there an abnormal region requiring risk assessment in this image? If so, please delineate it.",
  "Risk assessment of a thyroid nodule first requires locating the nodule and fully displaying its extent before analyzing its sonographic features. {EVID} Therefore, delineate the nodule itself. The boundary follows the interface between the nodule and the surrounding gland, excluding the whole thyroid, cervical vessels, and adjacent muscles."),
 "肝实质": ("Please delineate the region to observe when judging whether the liver has diffuse changes.",
  "Diffuse liver changes are assessed by observing representative parenchyma as a whole rather than a single focal lesion. {EVID} Therefore, delineate the continuous visible liver parenchyma, excluding definite focal lesions, major vessels and bile ducts, adjacent organs, and obvious artifact regions."),
 "右心房": ("Please delineate the region required for measuring the right atrial area.",
  "Right atrial area requires planar tracing along the inner edge of the right atrial blood pool at a specified view and phase, so the right atrial cavity must first be identified. {EVID} Therefore, delineate the blood cavity enclosed by the right atrial wall. The boundary follows the endocardial side and closes at the tricuspid annulus, excluding the right ventricle, the atrial wall, and connected great veins."),
 "左心房": ("Please delineate the region required for measuring the left atrial volume.",
  "Left atrial volume is usually estimated from the left atrial contour in an apical view by area-length or biplane methods, so the blood pool contour of a suitable view is required. {EVID} Therefore, delineate the cavity enclosed by the left atrial wall. The boundary follows the endocardial side and closes at the mitral annulus, excluding the left ventricle, the left atrial appendage, and the pulmonary vein extensions."),
 "结肠肠壁": ("Please delineate the region required for measuring the colonic wall thickness.",
  "Colonic wall thickness is measured perpendicular to the wall from the luminal to the outer interface on a recognizable wall segment, so the continuous bowel wall must first be identified. {EVID} Therefore, delineate the wall segment. The extent is bounded by the inner and outer wall interfaces, excluding luminal contents and perienteric tissue."),
 "胎儿腹围截面": ("Please delineate the region required for measuring the fetal abdominal circumference.",
  "Fetal abdominal circumference requires elliptical tracing along the outer abdominal wall on an appropriate transverse section, so whether the current plane meets the measurement requirement must first be judged. {EVID} Therefore, delineate the fetal abdominal cross-section. The boundary follows the outer abdominal wall, excluding fetal structures outside the abdomen and the surrounding amniotic fluid."),
 "耻骨联合": ("Please delineate the maternal structure serving as the fixed reference for measuring the angle of progression.",
  "The angle of progression is measured in a transperineal sagittal plane with the long axis of the pubic symphysis as the fixed reference, drawing the tangent from its lower edge to the fetal head. {EVID} Therefore, delineate the pubic symphysis region. The boundary follows its visible outer margin, excluding the fetal head and other adjacent hyperechoic interfaces."),
 "胎头": ("Please delineate the fetal part from which the tangent is determined for measuring the angle of progression.",
  "The angle of progression requires drawing a tangent from the lower edge of the pubic symphysis to the foremost skull contour of the fetal head, so the fetal head and its leading edge must first be identified. {EVID} Therefore, delineate the fetal head. The boundary follows the continuous skull contour, excluding the maternal pubic symphysis and surrounding soft tissue."),
 "颈总动脉": ("Please delineate the region required for measuring the common carotid artery diameter.",
  "Common carotid artery diameter is obtained by measuring the distance between the inner walls on both sides of the vessel, so the continuous lumen and its course must first be identified. {EVID} Therefore, delineate the lumen. The boundary follows the inner vessel wall interface, excluding the wall, adjacent veins, and other surrounding tissue."),
 "脐静脉": ("Please delineate the region to be located for fetal intrauterine transfusion.",
  "Intrauterine transfusion first requires locating an accessible umbilical vein and confirming its continuous course. {EVID} Therefore, delineate the visible umbilical vein lumen in the fetal abdomen. The boundary follows the inner vessel wall, excluding liver parenchyma, adjacent vessels, and extravascular tissue; the actual puncture site still requires dynamic scanning."),
 "前列腺腺体": ("Please delineate the region required for calculating the prostate volume.",
  "Prostate volume is usually estimated from the anteroposterior, transverse, and craniocaudal diameters on orthogonal planes, so the complete gland contour of each measurement plane is required. {EVID} Therefore, delineate the entire prostate gland visible in this plane. The boundary follows the capsule or the outer gland margin, excluding focal abnormalities and surrounding tissue."),
 "左心室腔": ("Please delineate the region required for calculating the ejection fraction.",
  "Ejection fraction compares the end-diastolic and end-systolic left ventricular volumes, so the endocardial contours at two key phases must be traced. {EVID} Therefore, delineate the left ventricular cavity enclosed by the myocardium. The boundary follows the endocardial side and closes at the mitral annulus, excluding the left atrium, the left ventricular myocardium, and regions beyond the annulus."),
 "甲状腺腺体": ("Please delineate the region required for calculating the thyroid volume.",
  "Thyroid volume is usually estimated from the length, width, and thickness of both lobes on orthogonal planes, so the complete outer gland contour is required. {EVID} Therefore, delineate the entire visible thyroid parenchyma. The boundary follows the gland margin; intraglandular nodules are not separate targets, and the trachea, cervical vessels, and adjacent muscles are excluded."),
 "受检肌肉区域": ("Please delineate the region required for evaluating whether the examined muscle is atrophic.",
  "Muscle atrophy is usually evaluated by muscle thickness, cross-sectional area, and echotexture on standard planes, so the extent of the whole examined muscle must be determined. {EVID} Therefore, delineate the muscle belly bounded by fascial or aponeurotic interfaces. The boundary follows the outer muscle margin, excluding single fascicles, adjacent subcutaneous tissue, and bone."),
 "右心室": ("Please delineate the region required for evaluating right ventricular size and systolic function.",
  "Right ventricular size and function are evaluated by chamber area and its change at a specified view and key phases, so the right ventricular endocardial contour must be traced. {EVID} Therefore, delineate the blood cavity enclosed by the right ventricular wall. The boundary follows the endocardial side and closes at the tricuspid annulus, excluding the right atrium, the right ventricular myocardium, and the left ventricle."),
 "左室心肌": ("Please delineate the region required for evaluating left ventricular myocardial thickness and motion.",
  "Left ventricular myocardial thickness and motion require comparing myocardial morphology across segments and phases, so the full extent of the myocardium must be determined. {EVID} Therefore, delineate the myocardial layer between the endocardium and epicardium surrounding the left ventricular cavity, excluding the left ventricular cavity, the left atrium, and extracardiac tissue."),
 "胎儿胃泡": ("Please delineate the abdominal structure to observe when evaluating fetal swallowing.",
  "Fetal swallowing cannot be judged from a single frame, but serial examinations observe whether the stomach is displayed and how it changes over time. {EVID} Therefore, delineate the intragastric fluid cavity. The boundary follows the inner gastric wall, excluding the fetal liver, intra-abdominal vessels, and other fluid structures."),
 "胎儿肝": ("Please delineate the organ to observe when evaluating fetal liver size and morphology.",
  "Ultrasound evaluation of the fetal liver requires displaying its contour to observe its size, morphology, and relation to intra-abdominal structures. {EVID} Therefore, delineate the visible liver parenchyma. The boundary follows the interface between the liver and surrounding tissue, excluding the stomach, major vessel lumens, and adjacent abdominal organs."),
 "胎头横断面区域": ("Please delineate the region required for measuring the fetal head circumference.",
  "Fetal head circumference requires tracing the outer head margin on an appropriate transverse section of the fetal head. {EVID} Therefore, delineate the complete fetal head cross-sectional region. The boundary follows the outer head margin, excluding tissue outside the fetal head."),
 "颈总动脉内中膜复合体": ("Please delineate the region required for measuring the intima-media thickness of the common carotid artery.",
  "Common carotid intima-media thickness is measured on a recognizable arterial wall segment, so the wall region to be measured must first be identified. {EVID} Therefore, delineate the intima-media complex band of the common carotid wall. The boundary follows its visible interfaces, excluding the vessel lumen and extravascular tissue."),
}
NEG_EN = {
 "乳腺阴性": ("Is there an abnormal region in this image that requires further evaluation? If so, please delineate it.",
  "Breast evaluation first requires identifying a focal lesion distinct from normal breast tissue. No focal target meeting this task definition is visible in this image, so no segmentation region is produced and an empty set is returned."),
 "肝脏阴性": ("Is there an abnormal region in the liver that requires further evaluation? If so, please delineate it.",
  "Liver evaluation first requires finding a focal lesion distinguishable from normal tissue in the parenchyma. No focal target meeting this task definition is visible in this image, so no segmentation region is produced and an empty set is returned."),
}
NEG_MAP = {"BrEaST": "乳腺阴性", "BUSI": "乳腺阴性", "AUL": "肝脏阴性"}

# ---------------- frozen V15 axes ----------------
_V17 = None

def _v17():
    """Frozen per-image evidence table, loaded on first use."""
    global _V17
    if _V17 is None:
        path = os.path.join(ROOT, "derived/reasoning_v17/reports/v17_density/V17_COT.jsonl")
        if not ROOT or not os.path.exists(path):
            raise FileNotFoundError(
                "V17 evidence table not found; set SONOINTENT_SOURCE_ROOT to the "
                "frozen source tree")
        _V17 = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                _V17[d["sample_id"]] = d
    return _V17

def gather_frozen(row):
    out = {}
    for u in row.get("reasoning_units") or []:
        if u.get("type") not in ("VISUAL_OBSERVATION", "DISCRIMINATION"):
            continue
        axis = u["semantic_key"][0]
        if axis in SP.FROZEN_AXES and axis not in out:
            out[axis] = u
    return out

def clause_frozen_en(axis, unit):
    direction = unit["semantic_key"][1]
    ref = re.sub(r"^its ", "", re.sub(r"^其", "", unit.get("ref") or "")).strip()
    REF_EN = {"紧邻的局部组织": "the adjacent tissue at the same depth",
              "同深度周围组织": "the surrounding tissue at the same depth"}
    ref = REF_EN.get(ref, "the adjacent tissue" if ref else "")
    if axis == "intensity_separability":
        if not ref:
            return None
        deg = "slightly" if unit.get("borderline") else "markedly"
        side = "lower" if direction == "lower" else "higher"
        return f"the interior echogenicity is {deg} {side} than {ref}"
    if axis == "posterior_acoustic_separability":
        return "posterior acoustic shadowing" if direction == "lower" else "posterior acoustic enhancement"
    return None

ALIAS_EN = {"左心室": "left ventricular cavity", "左心室腔": "left ventricular cavity",
            "左室心肌": "left ventricular myocardium", "左心房": "left atrium",
            "右心室": "right ventricle", "右心房": "right atrium",
            "胎头": "fetal head", "耻骨联合": "pubic symphysis",
            "胎儿肝": "fetal liver", "胎儿胃泡": "fetal stomach", "脐静脉": "umbilical vein",
            "肝实质": "liver parenchyma", "肝内占位": "focal liver lesion"}
def tgt_en(name):
    return TARGET_EN.get(name) or ALIAS_EN.get(name) or name

# ---------------- sibling masks ----------------
_MAN = None
_SIB = None

def _base(sid):
    return sid.rsplit("__t", 1)[0] if "__t" in sid else None

def _manifest():
    """Target manifest and sibling index, loaded on first use."""
    global _MAN, _SIB
    if _MAN is None:
        path = os.path.join(ROOT, "derived/reasoning_v15/manifest/sample_manifest.csv")
        if not ROOT or not os.path.exists(path):
            raise FileNotFoundError(
                "V15 sample manifest not found; set SONOINTENT_SOURCE_ROOT to the "
                "frozen source tree")
        with open(path, encoding="utf-8-sig") as f:
            _MAN = {r["sample_id"]: r for r in csv.DictReader(f)}
        _SIB = collections.defaultdict(list)
        for sid in _MAN:
            b = _base(sid)
            if b:
                _SIB[b].append(sid)
    return _MAN, _SIB

def sibling_evidence(sid, mask):
    man, sib = _manifest()
    out = {}
    for sm_sid in sib.get(_base(sid), []):
        if sm_sid == sid:
            continue
        other = man[sm_sid]["target_v14"]
        sm = np.asarray(Image.open(os.path.join(ROOT, man[sm_sid]["mask_path"]))) > 0
        if sm.shape != mask.shape:
            continue
        enc = M.sibling_enclosure(mask, sm)
        if "sibling_enclosure" not in out or enc["d_level"] > out["sibling_enclosure"]["rec"]["d_level"]:
            out["sibling_enclosure"] = {"rec": enc, "other": other}
        en2 = M.sibling_enclosure(sm, mask)
        if "encloses" not in out or en2["d_level"] > out["encloses"]["rec"]["d_level"]:
            out["encloses"] = {"rec": en2, "other": other}
        rp = M.relative_position(mask, sm)
        if "relative_position" not in out or abs(rp["depth_delta"]) > abs(out["relative_position"]["rec"]["depth_delta"]):
            out["relative_position"] = {"rec": rp, "other": other}
    return out

# ---------------- EN renderer ----------------
TARGET_EN = {"乳腺病灶":"breast lesion","消化道壁占位":"GI-wall lesion","肝内占位":"focal liver lesion",
 "卵巢肿块":"adnexal mass","甲状腺结节":"thyroid nodule","肝实质":"liver parenchyma","右心房":"right atrium",
 "左心房":"left atrium","结肠肠壁":"colonic wall","胎儿腹围截面":"fetal abdominal cross-section",
 "耻骨联合":"pubic symphysis","胎头":"fetal head","颈总动脉":"common carotid artery lumen",
 "脐静脉":"umbilical vein","前列腺腺体":"prostate gland","左心室腔":"left ventricular cavity",
 "甲状腺腺体":"thyroid gland","受检肌肉区域":"examined muscle","右心室":"right ventricle",
 "左室心肌":"left ventricular myocardium","胎儿胃泡":"fetal stomach","胎儿肝":"fetal liver",
 "胎头横断面区域":"fetal head cross-sectional region","颈总动脉内中膜复合体":"common carotid intima-media complex"}
OVERRIDE_EN = {
 ("前列腺腺体", "bright_interface_band", 2): "the gland margin shows a continuous hyperechoic interface",
 ("前列腺腺体", "bright_interface_band", 1): "the gland margin shows a partially visible hyperechoic interface",
 ("颈总动脉内中膜复合体", "elongation_axis", 2): "it extends continuously as a band along the vessel long axis",
 ("颈总动脉内中膜复合体", "elongation_axis", 1): "it extends as a band along the vessel long axis",
 ("胎头横断面区域", "bright_interface_band", 2): "the skull outer margin shows a continuous hyperechoic interface",
 ("胎头横断面区域", "bright_interface_band", 1): "the skull outer margin shows a partially visible hyperechoic interface",
 ("左心房", "ellipse_fit_quality", 2): "the contour is complete and regular",
 ("左心房", "ellipse_fit_quality", 1): "the contour is roughly regular",
 ("甲状腺腺体", "ellipse_fit_quality", 2): "the outer margin contour is continuously closed",
 ("甲状腺腺体", "ellipse_fit_quality", 1): "the outer margin contour is roughly continuous",
 ("甲状腺腺体", "elongation_axis", 1): "the visible gland is elongated along its long axis",
 ("甲状腺腺体", "elongation_axis", 2): "the visible gland shows marked extension along its long axis",
 ("前列腺腺体", "elongation_axis", 1): "the gland shows slight extension along its long axis on this slice",
 ("前列腺腺体", "elongation_axis", 2): "the gland shows clear extension along its long axis on this slice",
 ("耻骨联合", "elongation_axis", 1): "its long-axis orientation is roughly discernible",
 ("耻骨联合", "elongation_axis", 2): "its long-axis orientation is clear",
}
def clause_en(target, axis, rec, other=None):
    d = rec.get("d_level")
    if axis != "relative_position" and (d is None or d < 1):
        return None
    if (target, axis, d) in OVERRIDE_EN:
        return OVERRIDE_EN[(target, axis, d)]
    if axis == "anechoic_content":
        return "the interior is predominantly anechoic fluid" if d == 2 else "the interior is mostly hypoechoic to anechoic"
    if axis == "ellipse_fit_quality":
        return "the outer contour is continuous and regularly shaped" if d == 2 else "the outer contour is roughly continuous"
    if axis == "elongation_axis":
        return "it extends continuously along its course" if d == 2 else "it extends slightly along its course"
    if axis == "bright_interface_band":
        return "the outer margin shows a continuous hyperechoic interface" if d == 2 else "the outer margin shows a partially visible hyperechoic interface"
    if axis == "oriented_texture_coherence":
        return "the interior shows parallel striations" if d == 2 else "the interior shows a parallel striation tendency"
    if axis == "lumen_wall_ring":
        return "the wall echo encircles the lumen" if d == 2 else "the wall interface is partially visible"
    if axis == "parenchyma_proxy_uniformity":
        return "the surrounding gland appears uniform" if d == 2 else "the surrounding gland appears roughly uniform"
    if axis == "sibling_enclosure":
        return f"the boundary is mostly enclosed by the {tgt_en(other)}" if d == 2 else f"part of the boundary abuts the {tgt_en(other)}"
    if axis == "encloses":
        return f"it surrounds the {tgt_en(other)} along the boundary" if d == 2 else f"most of its boundary is adjacent to the {tgt_en(other)}"
    if axis == "relative_position":
        side = "shallower" if rec["direction"] == "shallower" else "deeper"
        return f"it lies {side} than the {tgt_en(other)}"
    return None

IMAGE_METRICS = {"anechoic_content", "bright_interface_band", "oriented_texture_coherence",
                 "lumen_wall_ring", "parenchyma_proxy_uniformity"}
MASK_METRICS = {"elongation_axis", "ellipse_fit_quality"}
SIBLING_METRICS = {"sibling_enclosure", "encloses", "relative_position"}

def select_evidence(target, img, mask, frozen, sib):
    prof = PROFILES[target]
    primary, forbidden = prof["primary"], set(prof.get("forbidden") or [])
    used, dropped = [], []
    suppress = set()
    ane = M.anechoic_content(img, mask)
    if (ane.get("d_level") or 0) >= 1:
        suppress.add("intensity_separability")
    for axis in primary:
        if axis in forbidden:
            dropped.append({"axis": axis, "reason": "forbidden_by_profile"}); continue
        if axis in suppress:
            dropped.append({"axis": axis, "reason": "suppressed_by_anechoic_content_same_meaning"}); continue
        if axis in IMAGE_METRICS:
            rec = getattr(M, axis)(img, mask); src = f"v19_metrics::{axis}"
            ev = {"metric": axis, "value": rec.get("value"), "d_level": rec.get("d_level"), "direction": rec.get("direction")}
            cl = clause_en(target, axis, rec)
        elif axis in MASK_METRICS:
            rec = getattr(M, axis)(mask); src = f"v19_metrics::{axis}"
            ev = {"metric": axis, "value": rec.get("value"), "d_level": rec.get("d_level"), "direction": rec.get("direction")}
            cl = clause_en(target, axis, rec)
        elif axis in SIBLING_METRICS:
            hit = sib.get(axis)
            if hit is None:
                dropped.append({"axis": axis, "reason": "no_sibling_mask_for_this_sample"}); continue
            rec, other = hit["rec"], hit["other"]
            src = f"v19_metrics::{axis}"
            ev = {"metric": axis, "value": rec.get("value"), "d_level": rec.get("d_level"),
                  "direction": rec.get("direction"), "other": other}
            cl = clause_en(target, axis, rec, other=other)
        else:
            u = frozen.get(axis)
            if u is None:
                dropped.append({"axis": axis, "reason": "no_frozen_v15_evidence_for_this_sample"}); continue
            src = "v15_frozen::" + (u.get("evidence_id") or "")
            ev = {"metric": axis, "value": None, "d_level": None, "direction": u["semantic_key"][1],
                  "frozen_validated": True}
            cl = clause_frozen_en(axis, u)
        if cl is None:
            dropped.append({"axis": axis, "d_level": ev.get("d_level"), "reason": "d0_below_threshold_or_no_direction"}); continue
        if len(used) >= 2:
            dropped.append({"axis": axis, "reason": "over_evidence_budget_max_2"}); continue
        used.append({"metric": axis, **ev, "primary": True, "forbidden": False,
                     "rendered_clause": cl, "source": src})
    return used, dropped


