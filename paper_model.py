"""paper_model - a VRM avatar as a few closed paper solids ("loft" method).

Each body part (head, torso, upper arm, ...) becomes a tube: rings of S corners at L levels along the bone,
closed by a flat cap at each end.  The radius of every corner is read from the avatar's surface, so hair, clothes
and shoes are inside the tube.  The colours are baked into one small map per part: (along the bone, around it)
for the sides and (across) for each cap.

Pure NumPy (and the textures from vrm_io).  Works in desktop Python and in the browser (Pyodide).
"""
# SPDX-License-Identifier: MIT
import math

import numpy as np

# part name, its humanoid bones, (from, to) bones for its axis, rings (levels), sides, the seam (where the band
# is cut open: an angle, 0 = the front / top) - the back for head and body, underneath for arms, inside for legs
PARTS = [
    ("HEAD", ["head", "leftEye", "rightEye", "jaw"], ("neck", "head"), 5, 8, 180),
    ("BODY", ["hips", "spine", "chest", "upperChest", "neck", "leftShoulder", "rightShoulder"],
     ("hips", "neck"), 6, 8, 180),
    ("L ARM", ["leftUpperArm", "leftLowerArm", "leftHand"], ("leftUpperArm", "leftMiddleDistal"), 5, 6, 90),
    ("R ARM", ["rightUpperArm", "rightLowerArm", "rightHand"], ("rightUpperArm", "rightMiddleDistal"), 5, 6, 270),
    ("L LEG", ["leftUpperLeg", "leftLowerLeg"], ("leftUpperLeg", "leftFoot"), 5, 6, 90),
    ("R LEG", ["rightUpperLeg", "rightLowerLeg"], ("rightUpperLeg", "rightFoot"), 5, 6, 270),
    ("L FOOT", ["leftFoot", "leftToes"], ("leftFoot", "leftToes"), 3, 6, 180),
    ("R FOOT", ["rightFoot", "rightToes"], ("rightFoot", "rightToes"), 3, 6, 180),
]
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")
SIDE_MAP = (48, 128)            # colour map of the sides: rows along the bone, columns around it
CAP_MAP = 64                    # colour map of a cap: square, across


def rest_points(model):
    """Every vertex in the rest pose (world space), its dominant node, uv and texture."""
    W = model.rest_world.astype(np.float32)
    out = []
    for p in model.parts:
        pos = np.c_[p["pos"], np.ones(len(p["pos"]), np.float32)]
        acc = np.zeros((len(pos), 3), np.float32)
        for k in range(4):
            node = p["skin_nodes"][p["slot"][:, k]]
            M = W[node] @ p["ibm"][p["slot"][:, k]] if len(p["ibm"]) > 1 else W[node] @ p["ibm"][0]
            acc += p["ww"][:, k:k + 1] * np.einsum("nij,nj->ni", M[:, :3, :], pos)
        dom = p["skin_nodes"][p["slot"][np.arange(len(pos)), p["ww"].argmax(1)]]
        out.append((acc, dom))
    return out


def part_of_nodes(model):
    """For every node: the paper part it belongs to (its nearest humanoid ancestor), or -1."""
    bone_part = {}
    for k, (_, bones, _, _, _, _) in enumerate(PARTS):
        for b in bones:
            if b in model.human:
                bone_part[model.human[b]] = k
    for side in ("left", "right"):                                  # fingers go to the hand (mittens)
        hand = PARTS.index(next(p for p in PARTS if p[0] == ("L ARM" if side == "left" else "R ARM")))
        for name, node in model.human.items():
            if name.startswith(side) and any(f in name for f in FINGERS):
                bone_part[node] = hand
    out = np.full(len(model.parent), -1, np.int64)
    for i in model.order:
        if i in bone_part:
            out[i] = bone_part[i]
        elif model.parent[i] >= 0:
            out[i] = out[model.parent[i]]
    return out


def surface_samples(model, pts, n=240_000, seed=3):
    """Points spread over the surface (by area), with their colour and dominant node."""
    rng = np.random.default_rng(seed)
    allp, allc, alln = [], [], []
    areas = []
    for p, (v, dom) in zip(model.parts, pts):
        tri = p["idx"].reshape(-1, 3)
        a = 0.5 * np.linalg.norm(np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]]), axis=1)
        areas.append(a)
    total = sum(a.sum() for a in areas) or 1.0
    for p, (v, dom), a in zip(model.parts, pts, areas):
        tri = p["idx"].reshape(-1, 3)
        if p["alpha"] == "BLEND" and len(tri) < 600:                  # eye highlights and such: skip
            continue
        cnt = rng.poisson(a / total * n).astype(np.intp)             # np.repeat wants the platform int (int32 in the browser)
        k = np.repeat(np.arange(len(tri)), cnt)
        if not len(k):
            continue
        r1, r2 = rng.random(len(k)), rng.random(len(k))
        s = np.sqrt(r1)
        b = np.stack([1 - s, s * (1 - r2), s * r2], 1).astype(np.float32)
        t3 = tri[k]
        xyz = (v[t3] * b[..., None]).sum(1)
        uv = (p["uv"][t3] * b[..., None]).sum(1)
        img = model.rgba[p["tex"]]
        h, w = img.shape[:2]
        px = np.clip((uv[:, 0] % 1.0) * w, 0, w - 1).astype(int)
        py = np.clip((uv[:, 1] % 1.0) * h, 0, h - 1).astype(int)
        col = img[py, px]
        keep = col[:, 3] >= 128
        nd = dom[t3[:, 0]]
        allp.append(xyz[keep])
        allc.append(col[keep, :3])
        alln.append(nd[keep])
    return np.concatenate(allp), np.concatenate(allc), np.concatenate(alln)


def joint(model, name):
    return model.rest_world[model.human[name]][:3, 3].astype(np.float64)


class PaperModel:
    """parts: list of dict(name, frame=(o, axis, u, v), verts (k, 3), faces [list of corner lists],
    kind ['side' | 'cap0' | 'cap1'], side_map (h, w, 3), cap_maps [2 x (c, c, 3)], t_range, r_max)."""

    def __init__(self, model, height_mm=180.0, min_edge_mm=5.0, n_samples=240_000):
        pts = rest_points(model)
        node_part = part_of_nodes(model)
        P, C, N = surface_samples(model, pts, n_samples)
        part = node_part[N]
        # VRM 0.x faces -Z: turn it to face +Z like VRM 1.0
        if model.version == "VRM 0.x":
            P = P * np.array([-1, 1, -1], np.float32)
            self.flip = np.array([-1.0, 1.0, -1.0])
        else:
            self.flip = np.array([1.0, 1.0, 1.0])
        self.height = float(P[:, 1].max() - P[:, 1].min())
        self.floor = float(P[:, 1].min())
        # thin arms and legs are made a little thicker, so that no edge of the paper is shorter than min_edge_mm
        self.min_edge = min_edge_mm / height_mm * self.height
        self.parts = []
        for k, (name, bones, (b0, b1), L, S, seam) in enumerate(PARTS):
            sel = part == k
            if sel.sum() < 50 or b0 not in model.human:
                continue
            q, col = P[sel].astype(np.float64), C[sel]
            o = joint(model, b0) * self.flip
            if b1 in model.human:
                axis = joint(model, b1) * self.flip - o
            else:
                axis = np.array([0.0, 1.0, 0.0])
            if name == "HEAD":
                axis = np.array([0.0, 1.0, 0.0])
            axis /= np.linalg.norm(axis)
            ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.8 else np.array([0.0, 1.0, 0.0])
            u = ref - axis * (ref @ axis)                              # theta = 0 looks forward (or up)
            u /= np.linalg.norm(u)
            v = np.cross(axis, u)
            piece = self.loft(name, q, col, o, axis, u, v, L, S)
            piece["seam"] = int(round(seam / 360 * S)) % S
            self.parts.append(piece)

    # ------------------------------------------------------------ one part
    def loft(self, name, q, col, o, axis, u, v, L, S):
        d = q - o
        t, x, y = d @ axis, d @ u, d @ v
        r = np.hypot(x, y)
        th = np.arctan2(y, x) % (2 * np.pi)
        t0, t1 = np.percentile(t, 0.3), np.percentile(t, 99.7)
        levels = np.linspace(t0, t1, L)
        dt = (t1 - t0) / (L - 1)
        sect = np.floor((th / (2 * np.pi)) * S + 0.5).astype(int) % S   # the corner nearest in angle
        R = np.zeros((L, S))
        for l in range(L):
            near = np.abs(t - levels[l]) <= dt * (0.5 if 0 < l < L - 1 else 0.75)
            for s in range(S):
                m = near & (sect == s)
                R[l, s] = np.percentile(r[m], 96) if m.sum() >= 4 else np.nan
        # holes: from the neighbours around the ring, then along the bone
        for l in range(L):
            row = R[l]
            if np.isnan(row).all():
                continue
            good = ~np.isnan(row)
            idx = np.arange(S)
            row[~good] = np.interp(idx[~good], idx[good], row[good], period=S)
        for s in range(S):
            col_ = R[:, s]
            good = ~np.isnan(col_)
            col_[~good] = np.interp(np.arange(L)[~good], np.arange(L)[good], col_[good]) if good.any() else 0.01
        R = np.maximum(R, 0.006)
        # a little smoothing around each ring (the paper does not like spikes)
        R = 0.6 * R + 0.2 * np.roll(R, 1, 1) + 0.2 * np.roll(R, -1, 1)
        R /= math.cos(math.pi / S)                                    # corners outside, edges through the surface
        R = np.maximum(R, self.min_edge / (2 * math.sin(math.pi / S)))  # no edge shorter than min_edge
        ang = np.arange(S) * 2 * np.pi / S
        verts = []
        for l in range(L):
            for s in range(S):
                verts.append(o + axis * levels[l] + R[l, s] * (math.cos(ang[s]) * u + math.sin(ang[s]) * v))
        verts = np.array(verts)
        faces, kind = [], []
        for l in range(L - 1):
            for s in range(S):
                a, b = l * S + s, l * S + (s + 1) % S
                c, d_ = (l + 1) * S + (s + 1) % S, (l + 1) * S + s
                faces += [[a, b, c], [a, c, d_]]                          # a quad as two triangles
                kind += ["side", "side"]
        faces.append(list(range(S)))
        kind.append("cap0")
        faces.append([(L - 1) * S + s for s in range(S)])
        kind.append("cap1")
        faces = [fix_winding(verts, f, o + axis * (t0 + t1) / 2) for f in faces]   # counter-clockwise from outside
        # colours
        side_map = bake_side(t, th, r, col, t0, t1)
        caps = [bake_cap(t, x, y, col, t0, dt, -1, R[0].max()), bake_cap(t, x, y, col, t1, dt, 1, R[-1].max())]
        return dict(name=name, frame=(o, axis, u, v), verts=verts, faces=faces, kind=kind,
                    side_map=side_map, cap_maps=caps, t_range=(t0, t1), levels=levels, R=R,
                    cap_r=(R[0].max(), R[-1].max()))

    # ------------------------------------------------------------ colour of any point on the paper solid
    def colour(self, k, pts3, face_kind):
        """RGB for 3D points lying on part k's faces (pts3 (n, 3)), all of one kind."""
        p = self.parts[k]
        o, axis, u, v = p["frame"]
        d = pts3 - o
        t, x, y = d @ axis, d @ u, d @ v
        if face_kind == "side":
            m = p["side_map"]
            h, w, _ = m.shape
            t0, t1 = p["t_range"]
            row = np.clip(((t - t0) / (t1 - t0) * h).astype(int), 0, h - 1)
            th = np.arctan2(y, x) % (2 * np.pi)
            c = np.clip((th / (2 * np.pi) * w).astype(int), 0, w - 1)
            return m[row, c]
        j = 0 if face_kind == "cap0" else 1
        m = p["cap_maps"][j]
        n = m.shape[0]
        rr = p["cap_r"][j] * 1.15
        ix = np.clip(((x / rr + 1) / 2 * n).astype(int), 0, n - 1)
        iy = np.clip(((y / rr + 1) / 2 * n).astype(int), 0, n - 1)
        return m[iy, ix]


def fix_winding(verts, f, inside):
    """Corners counter-clockwise seen from outside (normal pointing away from `inside`)."""
    p = verts[f]
    n = np.cross(p[1] - p[0], p[2] - p[0])
    if len(f) > 3:
        n = sum(np.cross(p[i] - p[0], p[i + 1] - p[0]) for i in range(1, len(f) - 1))
    if n @ (p.mean(0) - inside) < 0:
        return f[::-1]
    return f


def fill(m, have, wrap_x=True, steps=200):
    """Fill the empty texels of a colour map from their neighbours."""
    m = m.astype(np.float32)
    have = have.copy()
    for _ in range(steps):
        if have.all():
            break
        acc = np.zeros_like(m)
        cnt = np.zeros(have.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            hm = np.roll(have, (dy, dx), (0, 1))
            mm = np.roll(m, (dy, dx), (0, 1))
            if not wrap_x:
                if dx == 1:
                    hm[:, 0] = False
                if dx == -1:
                    hm[:, -1] = False
            if dy == 1:
                hm[0] = False
            if dy == -1:
                hm[-1] = False
            acc += mm * hm[..., None]
            cnt += hm
        new = (~have) & (cnt > 0)
        m[new] = acc[new] / cnt[new][:, None]
        have |= new
    return np.clip(m, 0, 255).astype(np.uint8)


def bake_side(t, th, r, col, t0, t1):
    """(along, around) map: in each texel the colour of the outermost point."""
    h, w = SIDE_MAP
    row = np.clip(((t - t0) / (t1 - t0) * h).astype(int), 0, h - 1)
    c = np.clip((th / (2 * np.pi) * w).astype(int), 0, w - 1)
    cell = row * w + c
    order = np.lexsort((r, cell))                                     # by cell, then by radius
    last = np.r_[cell[order][1:] != cell[order][:-1], True]
    m = np.zeros((h * w, 3), np.uint8)
    have = np.zeros(h * w, bool)
    m[cell[order][last]] = col[order][last]
    have[cell[order][last]] = True
    return fill(m.reshape(h, w, 3), have.reshape(h, w))


def bake_cap(t, x, y, col, t_end, dt, sign, rmax):
    """The cap's map: the points near the end, seen from outside along the bone."""
    n = CAP_MAP
    near = (t - t_end) * sign > -dt * 0.6
    if near.sum() < 10:
        near = np.ones(len(t), bool)
    rr = rmax * 1.15
    ix = np.clip(((x[near] / rr + 1) / 2 * n).astype(int), 0, n - 1)
    iy = np.clip(((y[near] / rr + 1) / 2 * n).astype(int), 0, n - 1)
    cell = iy * n + ix
    depth = t[near] * sign                                           # the outermost along the bone wins
    order = np.lexsort((depth, cell))
    last = np.r_[cell[order][1:] != cell[order][:-1], True]
    m = np.zeros((n * n, 3), np.uint8)
    have = np.zeros(n * n, bool)
    m[cell[order][last]] = col[near][order][last]
    have[cell[order][last]] = True
    return fill(m.reshape(n, n, 3), have.reshape(n, n), wrap_x=False)


def retro_copy(pm, colors=24, block=2):
    """The same paper model with retro colour maps: chunky texels and one small shared palette."""
    from PIL import Image
    import copy
    out = copy.copy(pm)
    out.parts = []
    maps = []
    for p in pm.parts:
        q = dict(p)
        sm = p["side_map"][::block, ::block]
        cm = [c[::block, ::block] for c in p["cap_maps"]]
        q["side_map"], q["cap_maps"] = sm, cm
        out.parts.append(q)
        maps += [sm.reshape(-1, 3)] + [c.reshape(-1, 3) for c in cm]
    allpix = np.concatenate(maps)
    side = int(math.ceil(math.sqrt(len(allpix))))
    sheet = np.zeros((side * side, 3), np.uint8)
    sheet[:len(allpix)] = allpix
    sheet[len(allpix):] = allpix[0]
    qimg = Image.fromarray(sheet.reshape(side, side, 3)).quantize(colors, method=Image.Quantize.MEDIANCUT,
                                                                  dither=Image.Dither.NONE)
    pal = np.array(qimg.getpalette()[:colors * 3], np.uint8).reshape(-1, 3)
    idx = np.asarray(qimg).reshape(-1)[:len(allpix)]
    flat = pal[idx]
    k = 0
    for q in out.parts:
        for key in ("side", 0, 1):
            m = q["side_map"] if key == "side" else q["cap_maps"][key]
            n = m.shape[0] * m.shape[1]
            new = flat[k:k + n].reshape(m.shape)
            k += n
            if key == "side":
                q["side_map"] = new
            else:
                q["cap_maps"] = list(q["cap_maps"])
                q["cap_maps"][key] = new
    out.palette = pal
    return out
