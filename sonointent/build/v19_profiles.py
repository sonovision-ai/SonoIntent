"""V19 per-target 推理画像。

每个 target 声明：哪些证据轴对这个器官有临床意义（primary，有序）、
哪些必须禁用（forbidden）、以及措辞用的名词。

设计原则：
  - primary 的顺序就是 CoT 里的陈述顺序 —— 器官特异的轴排在通用轴前面。
  - forbidden 是硬闸：即使 v15 冻结证据里有这条，V19 也不写。
    典型例子：前列腺腺体禁用 texture_separability（"颗粒感"不是前列腺的判据），
    心腔与管腔禁用 texture_separability（无回声腔谈"颗粒感"是把
    anechoic_content 的内容用错误的轴复述一遍）。
  - 不在 primary 也不在 forbidden 的轴 = secondary，仅在还有余额时补位。
"""

V15_AXES = {
    "intensity_separability",
    "texture_separability",
    "posterior_acoustic_separability",
    "rim_interior_separability",
    "boundary_gradient_separability",
    "adjacency_fraction",
}

# 每条 CoT 的事实预算（目标 75–95 字）
MAX_APPEARANCE_FACTS = 4
MAX_RELATION_FACTS = 3

_LESION = dict(
    kind="lesion",
    shape_noun="团块",
    primary=[
        "anechoic_content",            # 囊性 / 实性 —— 病灶最先要分的一刀
        "intensity_separability",      # 相对实质的回声强度
        "ellipse_fit_quality",         # 形态规则性
        "posterior_acoustic_separability",  # 后方效应
        "rim_interior_separability",   # 边缘 vs 内部
    ],
    forbidden=[],
)

PROFILES = {
    # ---------------- 占位 / 肿块类：BI-RADS 式四轴仍然是正统 ----------------
    "乳腺病灶": _LESION,
    "卵巢肿块": _LESION,
    "肝内占位": dict(_LESION, primary=[
        "anechoic_content", "intensity_separability", "ellipse_fit_quality",
        "posterior_acoustic_separability", "rim_interior_separability",
    ]),
    "消化道壁占位": dict(
        kind="lesion", shape_noun="团块",
        primary=["intensity_separability", "ellipse_fit_quality", "elongation_axis",
                 "posterior_acoustic_separability", "rim_interior_separability"],
        forbidden=[],
    ),

    # ---------------- 甲状腺 ----------------
    "甲状腺结节": dict(
        kind="lesion", shape_noun="结节",
        primary=["anechoic_content", "intensity_separability", "ellipse_fit_quality",
                 "posterior_acoustic_separability", "parenchyma_proxy_uniformity"],
        # TI-RADS 的五个维度里没有"颗粒感"；纹理轴在甲状腺上 VIP 早已判不成立
        forbidden=["texture_separability"],
    ),
    "甲状腺腺体": dict(
        kind="parenchyma", shape_noun="腺叶",
        primary=["intensity_separability", "elongation_axis", "texture_separability"],
        forbidden=[],
    ),

    # ---------------- 前列腺：整体形态与包膜，不是颗粒感 ----------------
    "前列腺腺体": dict(
        kind="gland", shape_noun="腺体",
        primary=["ellipse_fit_quality", "bright_interface_band",
                 "elongation_axis", "intensity_separability"],
        forbidden=["texture_separability"],
    ),

    # ---------------- 心腔：无回声腔 + 包绕关系 + 相邻腔室 ----------------
    "左心室腔": dict(
        kind="cavity", shape_noun="腔",
        primary=["anechoic_content", "sibling_enclosure", "relative_position",
                 "intensity_separability"],
        forbidden=["texture_separability"],
        enclosed_by=["左室心肌"],
    ),
    "左心房": dict(
        kind="cavity", shape_noun="腔",
        primary=["anechoic_content", "relative_position", "sibling_enclosure",
                 "intensity_separability"],
        forbidden=["texture_separability"],
        enclosed_by=[],
    ),
    "右心室": dict(
        kind="cavity", shape_noun="腔",
        primary=["anechoic_content", "relative_position", "sibling_enclosure",
                 "intensity_separability"],
        forbidden=["texture_separability"],
        enclosed_by=[],
    ),
    "右心房": dict(
        kind="cavity", shape_noun="腔",
        primary=["anechoic_content", "relative_position", "sibling_enclosure",
                 "intensity_separability"],
        forbidden=["texture_separability"],
        enclosed_by=[],
    ),
    "左室心肌": dict(
        kind="wall", shape_noun="心肌",
        primary=["intensity_separability", "encloses", "relative_position"],
        forbidden=[],
        encloses=["左心室腔", "左心室"],
    ),

    # ---------------- 血管：管腔 + 壁 + 走行 ----------------
    "颈总动脉": dict(
        kind="vessel", shape_noun="管腔",
        primary=["anechoic_content", "lumen_wall_ring", "elongation_axis",
                 "intensity_separability"],
        forbidden=["texture_separability"],
    ),
    "脐静脉": dict(
        kind="vessel", shape_noun="管腔",
        primary=["anechoic_content", "lumen_wall_ring", "elongation_axis",
                 "relative_position", "intensity_separability"],
        forbidden=["texture_separability"],
    ),

    # ---------------- 胎儿结构 ----------------
    "胎儿胃泡": dict(
        kind="fluid_space", shape_noun="液性腔",
        primary=["anechoic_content", "ellipse_fit_quality", "relative_position",
                 "intensity_separability"],
        forbidden=["texture_separability"],
    ),
    "胎儿肝": dict(
        kind="parenchyma", shape_noun="实质",
        primary=["intensity_separability", "relative_position", "texture_separability"],
        forbidden=[],
    ),
    "胎头": dict(
        kind="bony", shape_noun="颅骨轮廓",
        primary=["bright_interface_band", "ellipse_fit_quality", "relative_position",
                 "intensity_separability"],
        forbidden=[],
    ),
    "耻骨联合": dict(
        kind="bony", shape_noun="骨性结构",
        primary=["bright_interface_band", "relative_position", "elongation_axis",
                 "intensity_separability"],
        forbidden=[],
    ),
    "胎儿腹围截面": dict(
        kind="plane", shape_noun="截面",
        primary=["ellipse_fit_quality", "bright_interface_band", "intensity_separability"],
        forbidden=[],
    ),

    # ---------------- 肌肉 / 肠壁：定向纹理与带状延展 ----------------
    "受检肌肉区域": dict(
        kind="muscle", shape_noun="肌区",
        primary=["oriented_texture_coherence", "elongation_axis", "intensity_separability"],
        forbidden=[],
    ),
    "结肠肠壁": dict(
        kind="wall", shape_noun="肠壁",
        primary=["elongation_axis", "oriented_texture_coherence", "intensity_separability"],
        forbidden=[],
    ),

    # ---------------- 实质器官 ----------------
    "肝实质": dict(
        kind="parenchyma", shape_noun="实质",
        primary=["intensity_separability", "texture_separability", "relative_position"],
        forbidden=[],
    ),
}

# 需要 image+mask 的 v19 算子（按算子名给出调用签名类别）
V19_IMAGE_METRICS = {
    "anechoic_content", "oriented_texture_coherence", "bright_interface_band",
    "lumen_wall_ring", "parenchyma_proxy_uniformity",
}
V19_MASK_ONLY_METRICS = {"elongation_axis", "ellipse_fit_quality"}
V19_SIBLING_METRICS = {"sibling_enclosure", "relative_position", "encloses"}


def profile_for(target):
    return PROFILES.get(target)


def axis_rank(prof, axis):
    """primary 里的名次；不在 primary 的排到后面（secondary）。"""
    pri = prof["primary"]
    return pri.index(axis) if axis in pri else 100 + len(pri)


# ============================================================================
# V19.1 —— 依据《Ultrasound ReasonSeg: Target-specific CoT 描述规范》补入。
# core: 每条 CoT 的第一句，说明"要找的是什么"。它是任务/解剖层面的定义，
#       不含任何逐图断言，因此不需要 image evidence 支撑；逐图证据另起一句。
# no_prior: 规范明确禁止把"恒位于某结构深方"当核心的 target。
# ontology_unconfirmed: mask 语义未核实，本轮不得写死。
# ============================================================================

CORE = {
    "左心室腔": "本任务需要的是位于左室心肌内侧、由心肌包绕的血腔，分割边界对应心内膜侧",
    "左室心肌": "本任务需要的是构成左心室腔外围并包绕左室腔的心肌层，其内侧邻接心腔",
    "左心房": "本任务需要的是由左心房壁围成的腔内区域，应与左心室及心肌结构区分",
    "右心室": "本任务需要的是由心室壁围成的右心室血腔，在房室交界处与右心房分开",
    "右心房": "本任务需要的是右心房血腔，沿心房内膜侧边界划定，并在房室交界处与右心室分开",
    "颈总动脉": "本任务需要的是由血管壁限定的连续血管管腔，而不是管壁或邻近血管",
    "甲状腺腺体": "本任务需要的是整个甲状腺实质轮廓，应与结节、颈部血管及周围肌肉区分",
    "甲状腺结节": "本任务需要的是甲状腺实质内的局灶性结节，沿结节与周围腺体的界面划定范围",
    "前列腺腺体": "本任务需要的是完整的前列腺腺体外轮廓，覆盖可见腺体范围，而不是其中的局灶异常",
    "肝实质": "本任务需要的是连续的肝实质，用于评价整体肝组织，而不是肝内局灶病变",
    "肝内占位": "本任务需要的是肝内的局灶占位，沿病灶与宿主肝实质的界面划定范围",
    "乳腺病灶": "本任务需要的是独立于周围乳腺组织的局灶病灶，以病灶与周围组织的界面限定范围",
    "卵巢肿块": "本任务需要的是附件区与卵巢相关的局灶性占位，沿肿块外缘确定范围，而不是整个附件区",
    "消化道壁占位": "本任务需要的是与消化道壁相关的病灶，应体现其与壁的起源或附着关系，并与正常肠壁区分",
    "结肠肠壁": "本任务需要的是沿肠腔延伸的壁层结构，依据其连续走向及内外侧界面确定范围",
    "受检肌肉区域": "本任务需要的是由筋膜、腱膜界面限定的整块肌肉区域，而不是单条纹理",
    "胎头": "本任务需要的是胎儿头部的整体轮廓，依据连续弧形的颅骨界面识别，并与母体耻骨联合区分",
    "耻骨联合": "本任务需要的是母体的骨性参考结构，具有稳定的骨性界面与长轴，应与胎头的弧形轮廓区分",
    "胎儿腹围截面": "本任务需要的是用于腹围测量的胎儿腹部横断面区域",
    "胎儿肝": "本任务需要的是胎儿腹腔内的肝实质，结合其在胎儿腹部的位置以及与脐静脉等腹内结构的关系确定",
    "胎儿胃泡": "本任务需要的是胎儿腹部内相对独立的胃内液性腔，应与胎儿肝及血管结构区分",
    "脐静脉": "本任务需要的是连续的脐静脉管腔及其在胎儿腹部的走行，应与周围胎儿肝实质区分",
    "(negative)": "本例的冻结标注中没有对应的目标结构，因此输出空集",
}

# 规范「不应该把什么当核心」→ 禁止把语料级恒定深度关系写进这些 target
NO_PRIOR = {"左心房"}

# mask 语义未核实，核实前不得写死为"标准测量截面"
ONTOLOGY_UNCONFIRMED = {"胎儿腹围截面"}

# 实测失败、依 V18 纪律直接下架的轴（不调参、不复活）
RETIRED_AXES = {
    "parenchyma_proxy_uniformity": "14 条 13 条 D0，腺体实质代理不成立",
    "lumen_wall_ring": "13 条 11 条 D0，管壁亮脊在本语料上测不到",
}
for _t, _p in PROFILES.items():
    _p["primary"] = [a for a in _p["primary"] if a not in RETIRED_AXES]

# 规范补充的轴
PROFILES["左室心肌"]["primary"] = ["intensity_separability", "encloses",
                                   "elongation_axis", "relative_position"]
PROFILES["左心房"]["primary"] = ["anechoic_content", "sibling_enclosure",
                                 "intensity_separability"]
