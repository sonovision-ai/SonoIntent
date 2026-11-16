"""V19 器官特异性证据算子。

纪律（沿用 reasoning_v18/CLAUDE.md）：
  1. 每个算子先有合成自检（正例 / 反例），自检不过 → 不进语料。
  2. 不为了让指标"看起来有用"而调阈值。语料上大面积 D0 是合法结论。
  3. 不用一个指标代理它语义上不等价的概念。
  4. 全部确定性，无随机（自检里的噪声固定 seed）。

新算子一律新命名、新入口；未改动 v15 / v18 的任何指标实现。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

# ---------------------------------------------------------------- 常量
# 这些切点在实现时一次性设定，语料跑完不再回调（纪律 2）。
ANECHOIC_ABS = 40  # /255，无回声的绝对灰度上界；液性区几乎不产生回声
ANECHOIC_P90 = 80  # 真液性腔的 90 分位仍应偏暗

MIN_AREA_PX = 60  # 小于此面积的 mask 不做几何判定
RING_INNER, RING_OUTER = 2, 8  # 壁环带（紧贴 mask 外侧）
PROXY_INNER, PROXY_OUTER = 8, 25  # 实质代理环带（与 v15 local ring 同口径）
N_ANGLE_BINS = 72  # 角度分箱，5° 一格

__all__ = [
    "anechoic_content",
    "elongation_axis",
    "oriented_texture_coherence",
    "ellipse_fit_quality",
    "bright_interface_band",
    "lumen_wall_ring",
    "sibling_enclosure",
    "parenchyma_proxy_uniformity",
    "relative_position",
    "METRIC_IDS",
]

METRIC_IDS = [
    "anechoic_content",
    "elongation_axis",
    "oriented_texture_coherence",
    "ellipse_fit_quality",
    "bright_interface_band",
    "lumen_wall_ring",
    "sibling_enclosure",
    "parenchyma_proxy_uniformity",
    "relative_position",
]


# ---------------------------------------------------------------- 公共
def _pack(metric, value, d_level, direction, **extra):
    rec = {
        "metric": metric,
        "value": None if value is None else float(value),
        "d_level": d_level,
        "direction": direction,
        "reference_mode": extra.pop("reference_mode", "self_geometry_v1"),
        "measurement_version": "v19_" + metric + "_v1",
    }
    rec.update({k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
                for k, v in extra.items()})
    return rec


def _m(mask):
    return np.asarray(mask) > 127 if np.asarray(mask).dtype == np.uint8 else np.asarray(mask).astype(bool)


def _ring(mask, inner, outer):
    """mask 外侧 [inner, outer) px 的环带。"""
    d_in = ndi.binary_dilation(mask, iterations=inner)
    d_out = ndi.binary_dilation(mask, iterations=outer)
    return d_out & ~d_in


def _field(img):
    """成像扇区内的像素（排除黑边），用于归一化。"""
    return np.asarray(img) > 5


def _angular_bins(mask):
    """返回边界像素的角度分箱索引与其坐标。"""
    b = mask & ~ndi.binary_erosion(mask)
    ys, xs = np.nonzero(b)
    if ys.size == 0:
        return None, None, None
    cy, cx = ndi.center_of_mass(mask)
    ang = np.arctan2(ys - cy, xs - cx)
    idx = ((ang + np.pi) / (2 * np.pi) * N_ANGLE_BINS).astype(int) % N_ANGLE_BINS
    return idx, ys, xs


# ---------------------------------------------------------------- 1. 液性无回声
def anechoic_content(img, mask):
    """液性腔判据：mask 内低灰度占比 + 分布上界 + 内部离散度。

    语义 = 「该区域是否为无回声的液性腔」。不等价于 intensity_separability
    （后者是与参考区比较的相对强度，会把'比周围暗的实性低回声结节'也判为强）。
    """
    img = np.asarray(img).astype(np.float32)
    m = _m(mask)
    if m.sum() < MIN_AREA_PX:
        return _pack("anechoic_content", None, 0, "insufficient_area", n_px=int(m.sum()))
    v = img[m]
    dark_frac = float((v < ANECHOIC_ABS).mean())
    p90 = float(np.percentile(v, 90))
    iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
    fld = _field(img)
    rel = float(np.median(v) / max(np.median(img[fld]), 1.0)) if fld.sum() else float("nan")

    if dark_frac >= 0.60 and p90 <= ANECHOIC_P90:
        d, direction = 2, "anechoic"
    elif dark_frac >= 0.30:
        d, direction = 1, "partly_anechoic"
    else:
        d, direction = 0, "echogenic"
    return _pack("anechoic_content", dark_frac, d, direction,
                 p90=p90, iqr=iqr, rel_to_field=rel, n_px=int(m.sum()))


# ---------------------------------------------------------------- 2. 延展与走行
def elongation_axis(mask):
    """主轴长宽比与走行角度。语义 = 「管状/带状 还是 团块状」。"""
    m = _m(mask)
    if m.sum() < MIN_AREA_PX:
        return _pack("elongation_axis", None, 0, "insufficient_area", n_px=int(m.sum()))
    ys, xs = np.nonzero(m)
    pts = np.stack([xs - xs.mean(), ys - ys.mean()])
    cov = np.cov(pts)
    ev, evec = np.linalg.eigh(cov)
    lam2, lam1 = float(max(ev[0], 1e-9)), float(max(ev[1], 1e-9))
    elong = float(np.sqrt(lam1 / lam2))
    ax = evec[:, int(np.argmax(ev))]
    angle = float(np.degrees(np.arctan2(ax[1], ax[0])) % 180.0)
    # 走行方向：与探头横向(0°)/纵向(90°)的贴近程度
    orient = "longitudinal" if (angle < 30 or angle > 150) else ("transverse" if 60 < angle < 120 else "oblique")

    if elong >= 3.0:
        d, direction = 2, "tubular_band"
    elif elong >= 1.8:
        d, direction = 1, "elongated"
    else:
        d, direction = 0, "compact"
    return _pack("elongation_axis", elong, d, direction, axis_angle_deg=angle,
                 course=orient, n_px=int(m.sum()))


# ---------------------------------------------------------------- 3. 定向纹理
def oriented_texture_coherence(img, mask):
    """结构张量相干度 + 主方向集中度。语义 = 「是否呈平行条纹/束状排列」。

    明确不等价于 texture_separability（后者是局部方差的分布差异，
    对'有没有方向'完全不敏感）—— 这正是 V18 §4 指出的 catalog 缺口。
    """
    img = np.asarray(img).astype(np.float32)
    m = _m(mask)
    if m.sum() < MIN_AREA_PX:
        return _pack("oriented_texture_coherence", None, 0, "insufficient_area", n_px=int(m.sum()))
    gy, gx = np.gradient(ndi.gaussian_filter(img, 1.0))
    w = 2.0
    Jxx = ndi.gaussian_filter(gx * gx, w)
    Jyy = ndi.gaussian_filter(gy * gy, w)
    Jxy = ndi.gaussian_filter(gx * gy, w)
    tr = Jxx + Jyy
    det_term = np.sqrt(np.maximum((Jxx - Jyy) ** 2 + 4 * Jxy ** 2, 0))
    coh = np.where(tr > 1e-6, det_term / np.maximum(tr, 1e-6), 0.0)
    mean_coh = float(coh[m].mean())
    # 主方向集中度：倍角向量平均长度（圆统计），对 180° 周期正确
    theta2 = np.arctan2(2 * Jxy[m], (Jxx - Jyy)[m])
    wts = tr[m]
    if wts.sum() <= 1e-6:
        conc, dom = 0.0, float("nan")
    else:
        C = float((wts * np.cos(theta2)).sum() / wts.sum())
        S = float((wts * np.sin(theta2)).sum() / wts.sum())
        conc = float(np.hypot(C, S))
        dom = float((np.degrees(0.5 * np.arctan2(S, C)) + 90.0) % 180.0)

    if mean_coh >= 0.45 and conc >= 0.60:
        d, direction = 2, "parallel_striated"
    elif mean_coh >= 0.30 and conc >= 0.45:
        d, direction = 1, "weakly_oriented"
    else:
        d, direction = 0, "isotropic"
    return _pack("oriented_texture_coherence", mean_coh, d, direction,
                 orientation_concentration=conc, dominant_angle_deg=dom, n_px=int(m.sum()))


# ---------------------------------------------------------------- 4. 椭圆吻合
def ellipse_fit_quality(mask):
    """二阶矩椭圆与 mask 的 IoU。语义 = 「整体轮廓是否为规整的卵圆形」。"""
    m = _m(mask)
    if m.sum() < MIN_AREA_PX:
        return _pack("ellipse_fit_quality", None, 0, "insufficient_area", n_px=int(m.sum()))
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean(), xs.mean()
    pts = np.stack([xs - cx, ys - cy])
    cov = np.cov(pts)
    ev, evec = np.linalg.eigh(cov)
    ev = np.maximum(ev, 1e-9)
    # 等面积椭圆：半轴 = 2*sqrt(eigenvalue)（均匀椭圆的二阶矩关系）
    a = 2.0 * np.sqrt(ev)
    Y, X = np.nonzero(np.ones_like(m))
    d = np.stack([X - cx, Y - cy])
    proj = evec.T @ d
    inside = ((proj[0] / a[0]) ** 2 + (proj[1] / a[1]) ** 2) <= 1.0
    ell = inside.reshape(m.shape)
    inter = float((ell & m).sum())
    union = float((ell | m).sum())
    iou = inter / max(union, 1.0)
    n_comp = int(ndi.label(m)[1])

    if iou >= 0.90 and n_comp == 1:
        d_lv, direction = 2, "ovoid_regular"
    elif iou >= 0.80 and n_comp == 1:
        d_lv, direction = 1, "near_ovoid"
    else:
        d_lv, direction = 0, "irregular"
    return _pack("ellipse_fit_quality", iou, d_lv, direction, n_components=n_comp, n_px=int(m.sum()))


# ---------------------------------------------------------------- 5. 骨性亮带界面
def bright_interface_band(img, mask):
    """边界带亮度 z 分数 + 沿边界的连续覆盖。语义 = 「边界是否为连续的强回声界面」。

    与 rim_interior_separability 的区别：后者只比较边缘带与内部的灰度分布，
    对「这条亮带是一整条还是几段散点」完全不敏感（V18 §4 原话）。
    本算子的 longest_arc_frac 正是补这一点。
    """
    img = np.asarray(img).astype(np.float32)
    m = _m(mask)
    if m.sum() < MIN_AREA_PX:
        return _pack("bright_interface_band", None, 0, "insufficient_area", n_px=int(m.sum()))
    band = _ring(m, 0, 3) | (m & ~ndi.binary_erosion(m, iterations=3))
    surround = _ring(m, 6, 18)
    if band.sum() < 20 or surround.sum() < 50:
        return _pack("bright_interface_band", None, 0, "insufficient_band")
    s_med = float(np.median(img[surround]))
    s_sd = float(np.percentile(img[surround], 75) - np.percentile(img[surround], 25)) / 1.349
    s_sd = max(s_sd, 1.0)
    z = (float(np.median(img[band])) - s_med) / s_sd

    idx, ys, xs = _angular_bins(m)
    if idx is None:
        return _pack("bright_interface_band", z, 0, "no_boundary")
    thr = s_med + s_sd
    # 每个角度分箱：该方向的边界像素是否达到亮带阈值
    hit = np.zeros(N_ANGLE_BINS, dtype=bool)
    occ = np.zeros(N_ANGLE_BINS, dtype=bool)
    vals = img[ys, xs]
    for b in range(N_ANGLE_BINS):
        sel = idx == b
        if not sel.any():
            continue
        occ[b] = True
        hit[b] = float(np.percentile(vals[sel], 75)) >= thr
    n_occ = int(occ.sum())
    cov = float(hit[occ].mean()) if n_occ else 0.0
    # 最长连续弧（环形）
    if hit.any():
        двойн = np.concatenate([hit, hit])
        best = cur = 0
        for v in двойн:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        longest = min(best, N_ANGLE_BINS) / max(n_occ, 1)
    else:
        longest = 0.0
    longest = float(min(longest, 1.0))

    if z >= 1.0 and longest >= 0.50:
        d, direction = 2, "continuous_bright_interface"
    elif z >= 0.5 and cov >= 0.40:
        d, direction = 1, "partial_bright_interface"
    else:
        d, direction = 0, "no_bright_interface"
    return _pack("bright_interface_band", z, d, direction,
                 coverage=cov, longest_arc_frac=longest, n_bins_occupied=n_occ)


# ---------------------------------------------------------------- 6. 壁-腔关系
def lumen_wall_ring(img, mask):
    """紧贴 mask 外侧的环带是否成圈地构成一条**亮脊**。

    语义 = 「该区域是被一圈相对高回声的壁围绕的腔」。

    关键约束（自检里 NEG_hypoechoic_no_wall 逼出来的）：只要求"环带亮于 mask"
    是不够的 —— 任何低回声团块的周围组织都亮于它，那退化成 intensity 的变体。
    真正的壁必须同时亮于**更外侧的组织**，即在径向上是一条脊。
    两个条件都要在绝大多数角度方向上成立，才叫"成圈"。
    """
    img = np.asarray(img).astype(np.float32)
    m = _m(mask)
    if m.sum() < MIN_AREA_PX:
        return _pack("lumen_wall_ring", None, 0, "insufficient_area", n_px=int(m.sum()))
    ring = _ring(m, RING_INNER, RING_OUTER)
    outer = _ring(m, RING_OUTER + 2, RING_OUTER + 14)
    if ring.sum() < 60 or outer.sum() < 60:
        return _pack("lumen_wall_ring", None, 0, "insufficient_ring")
    lum = float(np.median(img[m]))
    wal = float(np.median(img[ring]))
    out = float(np.median(img[outer]))
    scale = max(float(np.median(img[_field(img)])), 1.0)
    margin = 0.05 * scale
    delta = (wal - max(lum, out)) / scale  # 相对两侧的脊高

    cy, cx = ndi.center_of_mass(m)

    def _bin(reg):
        ys, xs = np.nonzero(reg)
        ang = np.arctan2(ys - cy, xs - cx)
        return ((ang + np.pi) / (2 * np.pi) * N_ANGLE_BINS).astype(int) % N_ANGLE_BINS, img[ys, xs]

    rb, rv = _bin(ring)
    ob, ov = _bin(outer)
    bright = occ = 0
    for b in range(N_ANGLE_BINS):
        rs, os_ = rb == b, ob == b
        if rs.sum() < 3 or os_.sum() < 3:
            continue
        occ += 1
        w_b = float(np.median(rv[rs]))
        if w_b > lum + margin and w_b > float(np.median(ov[os_])) + margin:
            bright += 1
    frac = bright / max(occ, 1)

    if frac >= 0.85 and delta >= 0.10:
        d, direction = 2, "wall_encircles_lumen"
    elif frac >= 0.65 and delta >= 0.03:
        d, direction = 1, "partial_wall"
    else:
        d, direction = 0, "no_wall_lumen_pattern"
    return _pack("lumen_wall_ring", frac, d, direction, wall_lumen_delta=delta,
                 lumen_median=lum, wall_median=wal, outer_median=out, n_bins_occupied=occ,
                 reference_mode="perilesional_ring_v19")


# ---------------------------------------------------------------- 7. 被兄弟结构包绕
def sibling_enclosure(mask, sibling, radius=6):
    """target 周界中被 sibling 占据的**角度覆盖率**。

    与 adjacency_fraction（周界像素比例，切点 0.6/0.15）的区别：
    这里按角度分箱，回答的是"包绕了多少方向"而不是"接触了多少像素"，
    因此不复用 adjacency 的切点，另立切点并在此说明理由。
    """
    m, s = _m(mask), _m(sibling)
    if m.sum() < MIN_AREA_PX or s.sum() < MIN_AREA_PX:
        return _pack("sibling_enclosure", None, 0, "insufficient_area",
                     reference_mode="sibling_mask_v19")
    near = ndi.binary_dilation(s, iterations=radius)
    idx, ys, xs = _angular_bins(m)
    if idx is None:
        return _pack("sibling_enclosure", None, 0, "no_boundary", reference_mode="sibling_mask_v19")
    touch = near[ys, xs]
    occ = np.zeros(N_ANGLE_BINS, dtype=bool)
    hit = np.zeros(N_ANGLE_BINS, dtype=bool)
    for b in range(N_ANGLE_BINS):
        sel = idx == b
        if not sel.any():
            continue
        occ[b] = True
        hit[b] = bool(touch[sel].mean() >= 0.5)
    n_occ = int(occ.sum())
    cov = float(hit[occ].mean()) if n_occ else 0.0

    if cov >= 0.60:
        d, direction = 2, "largely_enclosed"
    elif cov >= 0.30:
        d, direction = 1, "partially_bordered"
    else:
        d, direction = 0, "not_enclosed"
    return _pack("sibling_enclosure", cov, d, direction, n_bins_occupied=n_occ,
                 radius_px=radius, reference_mode="sibling_mask_v19")


# ---------------------------------------------------------------- 8. 实质代理（甲状腺）
def parenchyma_proxy_uniformity(img, mask):
    """结节外周环带是否呈均一的腺体样回声（腺体实质的**代理**，非真值）。

    TN3K / DDTI 无腺体 mask。本算子不断言"结节位于腺体内"，
    只回答一个可测的、更弱的命题：结节紧邻的外周组织是否均一且与
    同深度带整体一致。措辞必须相应降级，且记录标 proxy_reference。
    """
    img = np.asarray(img).astype(np.float32)
    m = _m(mask)
    if m.sum() < MIN_AREA_PX:
        return _pack("parenchyma_proxy_uniformity", None, 0, "insufficient_area",
                     reference_mode="proxy_parenchyma_ring_v19", proxy_reference=True)
    ring = _ring(m, PROXY_INNER, PROXY_OUTER)
    if ring.sum() < 200:
        return _pack("parenchyma_proxy_uniformity", None, 0, "insufficient_ring",
                     reference_mode="proxy_parenchyma_ring_v19", proxy_reference=True)
    ys, _ = np.nonzero(m)
    y0, y1 = int(ys.min()), int(ys.max())
    band = np.zeros_like(m)
    band[y0:y1 + 1, :] = True
    band &= _field(img) & ~ndi.binary_dilation(m, iterations=PROXY_INNER)
    if band.sum() < 200:
        return _pack("parenchyma_proxy_uniformity", None, 0, "insufficient_band",
                     reference_mode="proxy_parenchyma_ring_v19", proxy_reference=True)

    rv, bv = img[ring], img[band]
    r_med = float(np.median(rv))
    scale = max(float(np.median(img[_field(img)])), 1.0)
    r_iqr = float(np.percentile(rv, 75) - np.percentile(rv, 25)) / scale
    agree = abs(r_med - float(np.median(bv))) / scale

    if r_iqr <= 0.30 and agree <= 0.15:
        d, direction = 2, "uniform_glandular_surround"
    elif r_iqr <= 0.45 and agree <= 0.25:
        d, direction = 1, "mostly_uniform_surround"
    else:
        d, direction = 0, "heterogeneous_surround"
    return _pack("parenchyma_proxy_uniformity", r_iqr, d, direction,
                 band_agreement=agree, ring_median=r_med,
                 reference_mode="proxy_parenchyma_ring_v19", proxy_reference=True)


# ---------------------------------------------------------------- 9. 相对空间位置
def relative_position(mask, sibling):
    """target 与 sibling 的质心深度差 / 侧向差（按图像尺寸归一）。

    单样本不下 D 级 —— 与 v15 scalar_pair 同口径，由语料级一致性裁决。
    """
    m, s = _m(mask), _m(sibling)
    if m.sum() < MIN_AREA_PX or s.sum() < MIN_AREA_PX:
        return _pack("relative_position", None, None, "insufficient_area",
                     reference_mode="sibling_mask_v19")
    my, mx = ndi.center_of_mass(m)
    sy, sx = ndi.center_of_mass(s)
    h, w = m.shape
    dd = float((my - sy) / h)
    dl = float((mx - sx) / w)
    return _pack("relative_position", dd, None, "deeper" if dd > 0 else "shallower",
                 lateral_delta=dl, depth_delta=dd,
                 area_log2_ratio=float(np.log2(max(m.sum(), 1) / max(s.sum(), 1))),
                 reference_mode="sibling_mask_v19")


# ================================================================ 合成自检
def _disc(shape, cy, cx, r):
    Y, X = np.ogrid[: shape[0], : shape[1]]
    return (Y - cy) ** 2 + (X - cx) ** 2 <= r * r


def _ellipse(shape, cy, cx, ry, rx, ang=0.0):
    Y, X = np.ogrid[: shape[0], : shape[1]]
    y, x = Y - cy, X - cx
    c, s = np.cos(ang), np.sin(ang)
    xr, yr = x * c + y * s, -x * s + y * c
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


def _selftest():
    rng = np.random.default_rng(42)
    S = (200, 200)
    rows = []
    ok = True

    def rec(group, case, r, expect):
        nonlocal ok
        got = r["d_level"]
        good = (got >= 2) if expect == "POS" else (got < 2)
        ok = ok and good
        rows.append((group, case, expect, r["value"], got, r["direction"], "OK" if good else "FAIL"))

    # --- anechoic_content
    base = (rng.normal(110, 18, S)).clip(0, 255)
    m = _disc(S, 100, 100, 30)
    img = base.copy(); img[m] = rng.normal(12, 5, m.sum()).clip(0, 255)
    rec("anechoic_content", "POS_fluid_cyst", anechoic_content(img, m), "POS")
    img2 = base.copy(); img2[m] = rng.normal(70, 20, m.sum()).clip(0, 255)
    rec("anechoic_content", "NEG_solid_hypoechoic", anechoic_content(img2, m), "NEG")
    img3 = base.copy(); img3[m] = rng.normal(170, 20, m.sum()).clip(0, 255)
    rec("anechoic_content", "NEG_hyperechoic", anechoic_content(img3, m), "NEG")

    # --- elongation_axis
    rec("elongation_axis", "POS_tube_6to1", elongation_axis(_ellipse(S, 100, 100, 12, 72)), "POS")
    rec("elongation_axis", "POS_band_oblique", elongation_axis(_ellipse(S, 100, 100, 10, 70, 0.6)), "POS")
    rec("elongation_axis", "NEG_disc", elongation_axis(_disc(S, 100, 100, 35)), "NEG")
    rec("elongation_axis", "NEG_ellipse_1_5", elongation_axis(_ellipse(S, 100, 100, 30, 45)), "NEG")

    # --- oriented_texture_coherence
    Y, X = np.mgrid[: S[0], : S[1]]
    stripes = (128 + 60 * np.sin(Y / 3.0)) + rng.normal(0, 6, S)
    roi = _ellipse(S, 100, 100, 30, 60)
    rec("oriented_texture_coherence", "POS_parallel_stripes",
        oriented_texture_coherence(stripes.clip(0, 255), roi), "POS")
    diag = (128 + 60 * np.sin((X + Y) / 4.0)) + rng.normal(0, 6, S)
    rec("oriented_texture_coherence", "POS_diagonal_stripes",
        oriented_texture_coherence(diag.clip(0, 255), roi), "POS")
    speck = rng.normal(120, 30, S).clip(0, 255)
    rec("oriented_texture_coherence", "NEG_isotropic_speckle",
        oriented_texture_coherence(speck, roi), "NEG")
    blobs = ndi.gaussian_filter(rng.normal(120, 60, S), 3).clip(0, 255)
    rec("oriented_texture_coherence", "NEG_isotropic_blobs",
        oriented_texture_coherence(blobs, roi), "NEG")

    # --- ellipse_fit_quality
    rec("ellipse_fit_quality", "POS_ellipse", ellipse_fit_quality(_ellipse(S, 100, 100, 35, 55)), "POS")
    rec("ellipse_fit_quality", "POS_disc", ellipse_fit_quality(_disc(S, 100, 100, 40)), "POS")
    star = np.zeros(S, bool)
    for k in range(7):
        a = 2 * np.pi * k / 7
        star |= _disc(S, 100 + 32 * np.sin(a), 100 + 32 * np.cos(a), 15)
    rec("ellipse_fit_quality", "NEG_star_lobulated", ellipse_fit_quality(star), "NEG")
    cres = _disc(S, 100, 100, 40) & ~_disc(S, 100, 118, 34)
    rec("ellipse_fit_quality", "NEG_crescent", ellipse_fit_quality(cres), "NEG")

    # --- bright_interface_band
    inner = _disc(S, 100, 100, 34)
    img = rng.normal(90, 15, S).clip(0, 255)
    shell = _disc(S, 100, 100, 38) & ~_disc(S, 100, 100, 32)
    img[shell] = rng.normal(225, 10, shell.sum()).clip(0, 255)
    img[_disc(S, 100, 100, 31)] = rng.normal(60, 10, _disc(S, 100, 100, 31).sum()).clip(0, 255)
    rec("bright_interface_band", "POS_full_bright_rim", bright_interface_band(img, inner), "POS")
    img_h = rng.normal(90, 15, S).clip(0, 255)
    half = shell & (X > 100)
    img_h[half] = rng.normal(225, 10, half.sum()).clip(0, 255)
    rec("bright_interface_band", "NEG_half_rim_only", bright_interface_band(img_h, inner), "NEG")
    rec("bright_interface_band", "NEG_uniform",
        bright_interface_band(rng.normal(110, 15, S).clip(0, 255), inner), "NEG")
    spots = np.zeros(S, bool)
    for k in range(6):
        a = 2 * np.pi * k / 6
        spots |= _disc(S, 100 + 35 * np.sin(a), 100 + 35 * np.cos(a), 4)
    img_s = rng.normal(90, 15, S).clip(0, 255)
    img_s[spots] = 235
    rec("bright_interface_band", "NEG_scattered_bright_spots", bright_interface_band(img_s, inner), "NEG")

    # --- lumen_wall_ring
    lum = _disc(S, 100, 100, 26)
    img = rng.normal(95, 15, S).clip(0, 255)
    wall = _disc(S, 100, 100, 33) & ~lum
    img[wall] = rng.normal(190, 12, wall.sum()).clip(0, 255)
    img[lum] = rng.normal(14, 5, lum.sum()).clip(0, 255)
    rec("lumen_wall_ring", "POS_vessel_wall_lumen", lumen_wall_ring(img, lum), "POS")
    img_n = rng.normal(95, 15, S).clip(0, 255)
    img_n[lum] = rng.normal(70, 12, lum.sum()).clip(0, 255)
    rec("lumen_wall_ring", "NEG_hypoechoic_no_wall", lumen_wall_ring(img_n, lum), "NEG")
    img_o = rng.normal(95, 15, S).clip(0, 255)
    oneside = wall & (X > 100)
    img_o[oneside] = 200
    img_o[lum] = rng.normal(60, 10, lum.sum()).clip(0, 255)
    rec("lumen_wall_ring", "NEG_one_sided_bright", lumen_wall_ring(img_o, lum), "NEG")

    # --- sibling_enclosure
    cav = _disc(S, 100, 100, 26)
    myo = _disc(S, 100, 100, 40) & ~_disc(S, 100, 100, 28)
    rec("sibling_enclosure", "POS_cavity_in_myocardium", sibling_enclosure(cav, myo), "POS")
    bar = np.zeros(S, bool); bar[132:150, 60:140] = True
    rec("sibling_enclosure", "NEG_adjacent_bar_only", sibling_enclosure(cav, bar), "NEG")
    far = _disc(S, 40, 40, 20)
    rec("sibling_enclosure", "NEG_far_structure", sibling_enclosure(cav, far), "NEG")
    halfmyo = myo & (X > 100)
    rec("sibling_enclosure", "NEG_half_ring", sibling_enclosure(cav, halfmyo), "NEG")

    # --- parenchyma_proxy_uniformity
    nod = _disc(S, 100, 100, 22)
    img = rng.normal(140, 10, S).clip(0, 255)      # 均一腺体样背景
    img[nod] = rng.normal(70, 12, nod.sum()).clip(0, 255)
    rec("parenchyma_proxy_uniformity", "POS_uniform_parenchyma",
        parenchyma_proxy_uniformity(img, nod), "POS")
    img_h = rng.normal(140, 10, S).clip(0, 255)
    blk = np.zeros(S, bool); blk[60:100, :] = True                   # 一半是别的组织
    img_h[blk] = rng.normal(45, 25, blk.sum()).clip(0, 255)
    img_h[nod] = rng.normal(70, 12, nod.sum()).clip(0, 255)
    rec("parenchyma_proxy_uniformity", "NEG_heterogeneous_surround",
        parenchyma_proxy_uniformity(img_h, nod), "NEG")

    w = max(len(r[1]) for r in rows) + 2
    print(f"{'group':30s}{'case':{w}s}{'exp':5s}{'value':>10s}{'D':>3s}  direction")
    print("-" * (46 + w))
    cur = None
    for g, c, e, v, d, di, st in rows:
        if g != cur:
            print(f"[{g}]")
            cur = g
        vs = "—" if v is None else f"{v:.4f}"
        print(f"{'':30s}{c:{w}s}{e:5s}{vs:>10s}{d:>3d}  {di:28s}{st}")
    print("-" * (46 + w))
    print("SELFTEST =", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
