# -*- coding: utf-8 -*-
"""Deterministic target identification for generated segmentation responses.

TargetAcc asks whether the generated reasoning names the annotated target, not
whether the mask overlaps it. The generated turn has one regular shape::

    <rationale>. In this image, <observation>. Therefore, delineate <TARGET>.
    The boundary follows <edge>, excluding <OTHER STRUCTURES>.

The target is the object of the last `delineate ...` clause, with the
`, excluding ...` tail cut before matching. Aliases are anchored at the start
of the object phrase and tried most-specific first, so a compound phrase wins
over a structure it mentions in passing. A response that does not resolve to a
known target is `unresolved` rather than guessed; no model or judge is used.
"""
import re

from .report import SLUG_ZH, DATASET_ZH, target_of  # noqa: F401  (re-exported)

# display name -> English slug, so a row carries both spellings
ZH_SLUG = {
    "胎儿腹围截面": "fetal_abdominal_cross-section",
    "肝内占位": "focal_liver_lesion",
    "左心房": "left_atrium",
    "左心室腔": "left_ventricular_cavity",
    "左室心肌": "left_ventricular_myocardium",
    "颈总动脉内中膜复合体": "common_carotid_intima-media_complex",
    "右心房": "right_atrium",
    "右心室": "right_ventricle",
    "胎儿肝": "fetal_liver",
    "胎儿胃泡": "fetal_stomach",
    "脐静脉": "umbilical_vein",
    "胎头": "fetal_head",
    "耻骨联合": "pubic_symphysis",
    "消化道壁占位": "gi-wall_lesion",
    "前列腺腺体": "prostate_gland",
    "甲状腺腺体": "thyroid_gland",
    "乳腺病灶": "breast_lesion",
    "甲状腺结节": "thyroid_nodule",
    "结肠肠壁": "colonic_wall",
    "颈总动脉": "common_carotid_artery_lumen",
    "胎头横断面区域": "fetal_head_cross-sectional_region",
    "卵巢肿块": "adnexal_mass",
    "受检肌肉区域": "examined_muscle",
}
# structure class, used only to decide whether a generic noun phrase ("the lesion
# itself") names the same kind of thing as the standard target.  The class is a
# property of the DATASET'S OWN annotation, not of the generated text.
CLASS = {
    "fetal_abdominal_cross-section": "region",
    "focal_liver_lesion": "lesion",
    "left_atrium": "cardiac_chamber",
    "left_ventricular_cavity": "cardiac_chamber",
    "left_ventricular_myocardium": "myocardium",
    "common_carotid_intima-media_complex": "vessel_wall",
    "right_atrium": "cardiac_chamber",
    "right_ventricle": "cardiac_chamber",
    "fetal_liver": "fetal_organ",
    "fetal_stomach": "fetal_organ",
    "umbilical_vein": "vessel",
    "fetal_head": "fetal_region",
    "pubic_symphysis": "bone",
    "gi-wall_lesion": "lesion",
    "prostate_gland": "gland",
    "thyroid_gland": "gland",
    "breast_lesion": "lesion",
    "thyroid_nodule": "nodule",
    "colonic_wall": "wall",
    "common_carotid_artery_lumen": "vessel_lumen",
    "fetal_head_cross-sectional_region": "fetal_region",
    "adnexal_mass": "mass",
    "examined_muscle": "muscle",
}

# ---------------------------------------------------------------------------
# 2. The frozen alias dictionary
# ---------------------------------------------------------------------------
# (pattern, slug, match_level).  Patterns are anchored at the START of the
# object phrase and tried in this order, so a specific compound phrase wins over
# a structure it mentions on the way.  `match_level` is "target" when the alias
# names the annotated structure itself, "class" when it names only the KIND of
# thing ("the lesion itself") and the verdict therefore depends on the dataset's
# own class -- those two are reported separately.
ALIAS = [
    # -- prostate / thyroid / breast / adnexa / GI -------------------------
    (r"entire prostate gland", "prostate_gland", "target"),
    (r"prostate gland", "prostate_gland", "target"),
    (r"thyroid gland", "thyroid_gland", "target"),
    (r"thyroid nodule", "thyroid_nodule", "target"),
    (r"breast lesion", "breast_lesion", "target"),
    (r"adnexal mass", "adnexal_mass", "target"),
    (r"common carotid intima[- ]media", "common_carotid_intima-media_complex", "target"),
    (r"carotid artery", "common_carotid_artery_lumen", "target"),
    (r"colonic wall", "colonic_wall", "target"),
    (r"examined muscle", "examined_muscle", "target"),
    (r"muscle", "examined_muscle", "target"),
    # -- cardiac -----------------------------------------------------------
    (r"myocardial layer", "left_ventricular_myocardium", "target"),
    (r"myocardium between the endocardium", "left_ventricular_myocardium", "target"),
    (r"myocardium", "left_ventricular_myocardium", "target"),
    (r"left ventricular cavity", "left_ventricular_cavity", "target"),
    (r"cavity enclosed by the left atrial wall", "left_atrium", "target"),
    (r"cavity enclosed by the left atrium", "left_atrium", "target"),
    (r"left atrial cavity", "left_atrium", "target"),
    (r"left atrium", "left_atrium", "target"),
    (r"blood cavity enclosed by the right atrial wall", "right_atrium", "target"),
    (r"blood cavity enclosed by the right atrium", "right_atrium", "target"),
    # "right ventricular wall" / "right ventricular myocardium" both name the
    # wall AROUND the right ventricular cavity, i.e. the cavity is the object.
    (r"blood cavity enclosed by the right ventricul", "right_ventricle", "target"),
    (r"cavity enclosed by the right ventricul", "right_ventricle", "target"),
    (r"cavity enclosed by the right atrial wall", "right_atrium", "target"),
    (r"cavity enclosed by the right atrium", "right_atrium", "target"),
    (r"right atrium", "right_atrium", "target"),
    (r"right ventricle", "right_ventricle", "target"),
    # "the cavity enclosed by the myocardium" -- the myocardium is named as the
    # BOUNDARY, the object is the cavity, so this is the LV cavity, not the wall.
    (r"blood cavity enclosed by the myocardium", "left_ventricular_cavity", "target"),
    (r"cavity enclosed by the myocardium", "left_ventricular_cavity", "target"),
    # -- fetal -------------------------------------------------------------
    (r"fetal abdominal cross[- ]section", "fetal_abdominal_cross-section", "target"),
    (r"fetal abdominal wall", "fetal_abdominal_wall", "target"),
    (r"fetal head cross[- ]sectional region", "fetal_head_cross-sectional_region", "target"),
    (r"fetal head", "fetal_head", "target"),
    (r"fetal liver", "fetal_liver", "target"),
    (r"liver parenchyma", "fetal_liver", "target"),
    (r"intragastric fluid cavity", "fetal_stomach", "target"),
    (r"fetal stomach", "fetal_stomach", "target"),
    (r"visible umbilical vein", "umbilical_vein", "target"),
    (r"umbilical vein", "umbilical_vein", "target"),
    (r"pubic symphysis", "pubic_symphysis", "target"),
    # -- generic classes (verdict depends on the dataset's own class) -------
    (r"lesion", "lesion", "class"),
    (r"nodule", "nodule", "class"),
    (r"mass", "mass", "class"),
    (r"abnormal region", "lesion", "class"),
    (r"wall segment", "wall", "class"),
    # -- vessel lumen ------------------------------------------------------
    # "the lumen" names a structure but not an organ, so the verdict comes from
    # the dataset's own class.  CCAUI is annotated `common_carotid_artery_lumen`
    # and CUBS `common_carotid_intima-media_complex`, and the two datasets get
    # byte-identical generated text -- so the same words come out correct on one
    # and wrong on the other, which is the whole point of the column.
    (r"lumen", "vessel_lumen", "class"),
]
ALIAS = [(re.compile(p), s, lv) for p, s, lv in ALIAS]

# Class-level aliases match when the dataset's own class is one of these.
CLASS_ALIASES = {"lesion", "nodule", "mass", "wall", "vessel_lumen"}

ANSWER_CUT = re.compile(r"\s*Sure, the segmentation result")
DELINEATE = re.compile(r"delineate\s+(?:the|a|an)\s+", re.I)


def pre_answer(text):
    """The generated turn with the answer template removed."""
    return ANSWER_CUT.split(text or "")[0].strip()


def _object_phrase(chunk):
    """Trim a delineate clause down to its object."""
    m = re.search(r"\.(?:\s|$)", chunk)          # first sentence end
    if m:
        chunk = chunk[:m.start()]
    chunk = re.split(r",\s*excluding\b", chunk, flags=re.I)[0]
    chunk = re.split(r"\s+without including\b", chunk, flags=re.I)[0]
    return chunk.strip().strip(",").strip()


def extract_target(text):
    """(raw object phrase, rule) or (None, rule).  Deterministic, no fallback
    to a model and no scoring of plausibility."""
    pre = pre_answer(text)
    hits = list(DELINEATE.finditer(pre))
    if hits:
        last = hits[-1]
        return _object_phrase(pre[last.end():]), "T1_delineate_clause"
    # No delineate clause.  The remaining rows open with a bare structure label
    # ("Right atrium. The boundary follows ...") -- use the first sentence, and
    # only when it is a bare noun phrase, never a rationale.
    first = re.split(r"[.!?](?:\s|$)", pre)[0].strip()
    if first and not re.search(r"\b(is|are|was|were|requires|require|follows|compares)\b",
                               first, re.I):
        return first, "T2_leading_label"
    return None, "U_no_cue"


LEAD = re.compile(r"^(?:the|a|an|entire|complete|whole|full|visible|its)\s+", re.I)


def resolve(raw):
    """(slug, match_level, alias) for an object phrase, or (None, None, None)."""
    if not raw:
        return None, None, None
    s = re.sub(r"\s+", " ", raw.strip().lower())
    # "the entire prostate gland", "the complete fetal head cross-sectional
    # region", "the visible liver parenchyma": the modifier is not part of the
    # structure, and dropping it here is why the table has one entry per target
    # instead of one per phrasing.
    while LEAD.match(s):
        s = LEAD.sub("", s, count=1)
    s = s.strip(" .;,")
    for pat, slug, level in ALIAS:
        if pat.match(s):
            return slug, level, pat.pattern
    return None, None, None


def classify(text, canon_slug):
    """-> dict(label, extracted, slug, match_level, rule, alias)."""
    raw, rule = extract_target(text)
    slug, level, alias = resolve(raw)
    if slug is None:
        return dict(label="unresolved", extracted=raw, slug=None,
                    match_level=None, rule=rule, alias=alias)
    hit = False
    if slug == canon_slug:
        hit = True
    elif level == "class" and slug in CLASS_ALIASES and CLASS.get(canon_slug) == slug:
        hit = True
    return dict(label="correct" if hit else "incorrect",
                extracted=raw, slug=slug, match_level=level,
                rule=rule, alias=alias)


