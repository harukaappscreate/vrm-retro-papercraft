"""paper_layout - unfold the paper solids of paper_model, add glue tabs and numbers, lay them out on A4 pages
and write a PDF.

- Unfolding: each part grows pieces face by face across shared edges (rows of the tube first, so the sides
  unroll as bands), placing each face flat next to its neighbour.  A face that would overlap starts a new piece.
- Overlaps: separating-axis tests on convex pieces (caps are split into triangles from their centre).
- Glue tabs: one per cut edge, trapezoids whose ends are cut back where a neighbour face is close.
- Paper: y runs down, so the layout is mirrored left-right before drawing (otherwise the pictures read backwards).
- Lines: cut = solid, mountain fold = dashed, valley fold = dash-dot.

Pure NumPy + Pillow.  Works in desktop Python and in the browser (Pyodide).
"""
# SPDX-License-Identifier: MIT
import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

A4 = (210.0, 297.0)
MARGIN = 8.0                    # mm
HEADER = 15.0                   # mm, the title and two lines of notes on every page
GAP = 3.0                       # mm between pieces
TAB_MM = 5.0                    # glue tab height
FLAT_DEG = 2.0                  # folds flatter than this get no line


# ---------------------------------------------------------------- geometry helpers
def convex_overlap(a, b, eps=1e-7):
    for poly in (a, b):
        k = len(poly)
        for i in range(k):
            e = poly[(i + 1) % k] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            tol = eps * (abs(n).sum() + 1)
            if pa.max() <= pb.min() + tol or pb.max() <= pa.min() + tol:
                return False
    return True


def convex_parts(poly2):
    """A face as convex pieces: triangles stay, polygons become a fan from their centre (caps are star-shaped)."""
    if len(poly2) == 3:
        return [poly2]
    c = poly2.mean(0)
    return [np.array([c, poly2[i], poly2[(i + 1) % len(poly2)]]) for i in range(len(poly2))]


def flatten(P3):
    """3D corners of a flat face -> 2D in its own plane, counter-clockwise when seen from outside."""
    o = P3[0]
    u = P3[1] - o
    u /= np.linalg.norm(u)
    n = np.zeros(3)
    for i in range(1, len(P3) - 1):
        n += np.cross(P3[i] - o, P3[i + 1] - o)
    n /= np.linalg.norm(n)
    w = np.cross(n, u)
    return np.stack([(P3 - o) @ u, (P3 - o) @ w], 1), n


def place(loc, ia, ib, pa, pb):
    """Move a flat face (local 2D) rigidly so that its corner ia lands on pa and ib on pb."""
    d0 = loc[ib] - loc[ia]
    d1 = pb - pa
    a = math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0])
    c, s = math.cos(a), math.sin(a)
    Rm = np.array([[c, -s], [s, c]])
    return (loc - loc[ia]) @ Rm.T + pa


# ---------------------------------------------------------------- unfolding one part
def unfold_part(part):
    """A tube as bands: each band (a ring of quads, two triangles each) unrolls as one strip, cut open at the
    seam; the widest band is the spine and the others hang off it at the middle of the strip (the side opposite
    the seam); the caps hang off the end bands.  Faces that would overlap start further pieces.
    -> list of pieces; a piece = dict(faces {face: 2D corners}, tree [(parent, child, edge)])."""
    V, F = part["verts"], part["faces"]
    L, S = part["R"].shape
    nf = len(F)
    loc, nrm = [], []
    for f in F:
        l2, n = flatten(V[f])
        loc.append(l2)
        nrm.append(n)
    edges = {}
    for k, f in enumerate(F):
        for i in range(len(f)):
            e = tuple(sorted((f[i], f[(i + 1) % len(f)])))
            edges.setdefault(e, []).append(k)

    def other(k, e):
        fs = edges[e]
        return fs[0] if fs[1] == k else fs[1]

    def tri(band, s, half):                                           # face number of a side triangle
        return band * 2 * S + 2 * s + half
    cap0, cap1 = nf - 2, nf - 1
    seam = part.get("seam", 0)
    mid = (seam + S // 2) % S
    widths = part["R"].mean(1)
    spine = int(np.argmax(widths[:-1] + widths[1:]))
    pieces, left = [], set(range(nf))

    def new_piece(root):
        flat = {root: loc[root].copy()}
        pc = dict(faces=flat, tree=[], cv={root: convex_parts(flat[root])})
        left.discard(root)
        pieces.append(pc)
        return pc

    def attach(pc, k, j, e):
        """Place face j next to face k (in piece pc) across edge e, if it fits."""
        if j not in left or k not in pc["faces"]:
            return False
        fk, fj = F[k], F[j]
        pa, pb = pc["faces"][k][fk.index(e[0])], pc["faces"][k][fk.index(e[1])]
        cand = place(loc[j], fj.index(e[0]), fj.index(e[1]), pa, pb)
        cv = convex_parts(cand)
        for q in pc["cv"].values():
            for x in q:
                for y in cv:
                    if convex_overlap(x, y):
                        return False
        pc["faces"][j] = cand
        pc["cv"][j] = cv
        pc["tree"].append((k, j, e))
        left.discard(j)
        return True

    def shared(a, b):
        sa, sb = set(F[a]), set(F[b])
        c = sorted(sa & sb)
        return tuple(c) if len(c) == 2 else None

    def grow_band(pc, band, start):
        """From triangle `start`, along the band both ways, stopping at the seam."""
        order = []
        for s in range(S):                                            # the strip's order: seam+1 ... seam
            q = (seam + 1 + s) % S
            order += [tri(band, q, 1), tri(band, q, 0)]                   # (a,c,d) then (a,b,c): the diagonal, then the next quad
        i0 = order.index(start)
        for direction in (1, -1):
            i = i0
            while 0 <= i + direction < len(order):
                a, b = order[i], order[i + direction]
                e = shared(a, b)
                if e is None or not attach(pc, a, b, e):
                    break
                i += direction

    def hang_band(pc, band, from_band):
        """Hang a band off a placed neighbour band at the quad opposite the seam (or the nearest that fits)."""
        for d in sorted(range(S), key=lambda x: min(x, S - x)):
            for sgn in (1, -1):
                s = (mid + sgn * d) % S
                for half_from in (0, 1):
                    for half_to in (0, 1):
                        a, b = tri(from_band, s, half_from), tri(band, s, half_to)
                        e = shared(a, b)
                        if e and a in pc["faces"] and attach(pc, a, b, e):
                            grow_band(pc, band, b)
                            return True
        return False

    pc = new_piece(tri(spine, mid, 0))
    grow_band(pc, spine, tri(spine, mid, 0))
    for band in list(range(spine + 1, L - 1)):
        hang_band(pc, band, band - 1)
    for band in list(range(spine - 1, -1, -1)):
        hang_band(pc, band, band + 1)
    for cap, band in ((cap0, 0), (cap1, L - 2)):
        for d in sorted(range(S), key=lambda x: min(x, S - x)):
            done = False
            for half in (0, 1):
                a = tri(band, (mid + d) % S, half)
                e = shared(a, cap)
                if e and attach(pc, a, cap, e):
                    done = True
                    break
            if done:
                break
    # whatever did not fit: more pieces, grown greedily (same band first)
    while left:
        root = min(left)
        pc = new_piece(root)
        front = [root]
        while front:
            k = front.pop(0)
            for i in range(len(F[k])):
                e = tuple(sorted((F[k][i], F[k][(i + 1) % len(F[k])])))
                if len(edges[e]) == 2:
                    j = other(k, e)
                    if attach(pc, k, j, e):
                        front.append(j)
    for p in pieces:
        p["normals"] = {k: nrm[k] for k in p["faces"]}
        del p["cv"]
    return pieces, edges


def bbox_area(piece):
    a = np.concatenate(list(piece["faces"].values()))
    w, h = a.max(0) - a.min(0)
    return w * h


# ---------------------------------------------------------------- glue tabs
def tab_shape(pa, pb, inside, hh, ra, rb):
    d = pb - pa
    L = np.linalg.norm(d)
    d = d / L
    n = np.array([-d[1], d[0]])
    if n @ (inside - pa) > 0:
        n = -n
    if (ra + rb) * hh > 0.9 * L:
        return None
    return np.array([pa, pb, pb - d * hh * rb + n * hh, pa + d * hh * ra + n * hh])


def add_tabs(pieces_all, part_edges, scale_mm):
    """One tab for every cut edge: on whichever of its two faces has room.  Numbers pair tab and edge."""
    hh0 = TAB_MM / scale_mm                                           # tab height in model units
    shapes = [(1.0, 1.0, 1.0), (0.75, 1.0, 1.0), (0.6, 3.5, 1.0), (0.6, 1.0, 3.5), (0.45, 3.5, 3.5),
              (0.35, 1.2, 1.2), (0.25, 2.5, 2.5), (0.18, 2.0, 2.0)]
    # where is every face: (part, piece)
    where = {}
    for pi, (part_k, piece) in enumerate(pieces_all):
        for f in piece["faces"]:
            where[(part_k, f)] = pi
        piece["tabs"] = []
        piece["marks"] = []
        piece["cv"] = [x for q in piece["faces"].values() for x in convex_parts(q)]
    num = 0
    for part_k, edges in part_edges:
        for e, fs in sorted(edges.items()):
            if len(fs) != 2:
                continue
            a, b = fs
            pa_i, pb_i = where[(part_k, a)], where[(part_k, b)]
            tree_edge = pa_i == pb_i and any(t[2] == e for t in pieces_all[pa_i][1]["tree"])
            if tree_edge:
                continue
            num += 1
            placed = None
            for h, ra, rb in shapes:
                for f_tab, f_mark in ((a, b), (b, a)):
                    pc = pieces_all[where[(part_k, f_tab)]][1]
                    poly = pc["faces"][f_tab]
                    face = pieces_all[where[(part_k, f_tab)]][1]
                    F = part_faces[part_k][f_tab]
                    p0, p1 = poly[F.index(e[0])], poly[F.index(e[1])]
                    t = tab_shape(p0, p1, poly.mean(0), hh0 * h * min(1.0, np.linalg.norm(p1 - p0) / hh0 / 1.4 + 0.3), ra, rb)
                    if t is None:
                        continue
                    if any(convex_overlap(t, x) for x in pc["cv"]) or any(convex_overlap(t, x[0]) for x in pc["tabs"]):
                        continue
                    placed = (pc, t, f_tab, f_mark, e)
                    break
                if placed:
                    break
            if placed is None:                                         # no room: the edge is glued with tape
                placed = (None, None, a, b, e)
            pc, t, f_tab, f_mark, e = placed
            if pc is not None:
                pc["tabs"].append((t, num))
            mc = pieces_all[where[(part_k, f_mark)]][1]
            mc["marks"].append((f_mark, e, num))
            if pc is None:
                tc = pieces_all[where[(part_k, f_tab)]][1]
                tc["marks"].append((f_tab, e, num))
    return num


part_faces = {}


# ---------------------------------------------------------------- the whole layout
def layout(pm, height_mm=150.0, seed=1):
    """-> dict(pages=[[placed piece, ...], ...], scale (mm per model unit), pieces, n_tabs)."""
    scale = height_mm / pm.height
    pieces_all, part_edges = [], []
    part_faces.clear()
    for k, part in enumerate(pm.parts):
        part_faces[k] = part["faces"]
        pieces, edges = unfold_part(part)
        for pc in pieces:
            pieces_all.append((k, pc))
        part_edges.append((k, edges))
    n_tabs = add_tabs(pieces_all, part_edges, scale)
    # turn each piece for a small box, then shelves
    boxes = []
    for i, (k, pc) in enumerate(pieces_all):
        pts = np.concatenate(list(pc["faces"].values()) + [t for t, _ in pc["tabs"]]) * scale
        best = None
        for deg in range(0, 180, 5):
            a = math.radians(deg)
            Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            q = pts @ Rm.T
            w, h = q.max(0) - q.min(0)
            if w > A4[0] - 2 * MARGIN:
                continue
            key = (h, w * h)
            if best is None or key < best[0]:
                best = (key, deg, q.min(0), w, h)
        if best is None:
            raise ValueError("a piece is wider than the page: make the figure smaller")
        _, deg, lo, w, h = best
        if h > A4[1] - 2 * MARGIN - HEADER:
            raise ValueError("a piece is taller than the page: make the figure smaller")
        boxes.append(dict(i=i, part=k, piece=pc, deg=deg, lo=lo, w=w, h=h))
    # first-fit shelves over all pages, tallest pieces first
    boxes.sort(key=lambda b: -b["h"])
    top, bottom, right = MARGIN + HEADER, A4[1] - MARGIN, A4[0] - MARGIN
    pages, shelves = [], []                                           # shelf: [page, y, h, x]
    for b in boxes:
        spot = None
        for sh in shelves:
            if b["h"] <= sh[2] + 1e-6 and sh[3] + b["w"] <= right:
                spot = sh
                break
        if spot is None:
            for pi in range(len(pages)):
                ys = [sh[1] + sh[2] + GAP for sh in shelves if sh[0] == pi]
                y = max(ys) if ys else top
                if y + b["h"] <= bottom:
                    spot = [pi, y, b["h"], MARGIN]
                    shelves.append(spot)
                    break
        if spot is None:
            pages.append([])
            spot = [len(pages) - 1, top, b["h"], MARGIN]
            shelves.append(spot)
        b["at"] = (spot[3], spot[1])
        spot[3] += b["w"] + GAP
        pages[spot[0]].append(b)
    return dict(pages=pages, scale=scale, n_tabs=n_tabs, n_pieces=len(pieces_all))


def to_page(b, pts, scale):
    """Model 2D -> page mm (turned, moved, then mirrored left-right inside its box)."""
    a = math.radians(b["deg"])
    Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    q = (pts * scale) @ Rm.T - b["lo"]
    q[:, 0] = b["w"] - q[:, 0]                                        # the mirror: y runs down on paper
    return q + np.array(b["at"])


# ---------------------------------------------------------------- drawing
def font(px):
    """(font, scale): Pillow's scalable default font, or (in a Pillow without FreeType) its small bitmap font
    drawn bigger."""
    try:
        return ImageFont.load_default(size=px), 1
    except Exception:
        return ImageFont.load_default(), max(1, round(px / 11))


def text_box(xy, s, f, anchor="la"):
    fo, k = f
    if k == 1:
        return ImageDraw.Draw(Image.new("L", (1, 1))).textbbox(xy, s, font=fo, anchor=anchor)
    l, t, r, b = fo.getbbox(s)
    w, h = (r - l) * k, (b - t) * k
    x, y = xy
    x -= {"l": 0, "m": w / 2, "r": w}[anchor[0]]
    y -= {"a": 0, "t": 0, "m": h / 2, "b": h, "s": h}[anchor[1]]
    return (x, y, x + w, y + h)


def put(img, xy, s, f, fill, anchor="la"):
    """Text on the page, in either kind of font."""
    fo, k = f
    if k == 1:
        ImageDraw.Draw(img).text(xy, s, fill=fill, font=fo, anchor=anchor)
        return
    l, t, r, b = fo.getbbox(s)
    m = Image.new("L", (max(1, r - l), max(1, b - t)), 0)
    ImageDraw.Draw(m).text((-l, -t), s, fill=255, font=fo)
    m = m.resize((m.width * k, m.height * k), Image.NEAREST)
    x0, y0, _, _ = text_box(xy, s, f, anchor)
    img.paste(Image.new("RGB", m.size, fill), (int(x0), int(y0)), m)


def render_pages(pm, lay, dpi=200, title="VRM RETRO PAPERCRAFT", note="", retro=None, progress=None):
    """-> list of RGB page images."""
    mm = dpi / 25.4
    W, H = round(A4[0] * mm), round(A4[1] * mm)
    out = []
    lw = max(1, round(0.2 * mm))
    f_small, f_mid, f_big = font(round(2.2 * mm)), font(round(2.6 * mm)), font(round(4.0 * mm))
    for pn, page in enumerate(lay["pages"]):
        if progress:
            progress(pn, len(lay["pages"]))
        img = Image.new("RGB", (W, H), "white")
        a = np.asarray(img).copy()
        # glue tabs first (light grey), then the faces: once with a bleed over the cut edges, once exactly
        tab_polys = []
        for b in page:
            for t, num in b["piece"]["tabs"]:
                tab_polys.append((to_page(b, t, lay["scale"]) * mm, num))
        im2 = Image.fromarray(a)
        d2 = ImageDraw.Draw(im2)
        for p, num in tab_polys:
            d2.polygon([tuple(x) for x in p], fill=(232, 232, 232))
        a = np.asarray(im2).copy()
        for bleed in (0.8 * mm, 0.0):
            for b in page:
                part = pm.parts[b["part"]]
                for f, poly in b["piece"]["faces"].items():
                    q2 = to_page(b, poly, lay["scale"]) * mm
                    P3 = part["verts"][part["faces"][f]]
                    paint_face(a, q2, P3, pm, b["part"], part["kind"][f], bleed, retro)
        img = Image.fromarray(a)
        d = ImageDraw.Draw(img)

        def dashed(pa, pb, pattern, colr):
            L = float(np.linalg.norm(pb - pa))
            if L < 1:
                return
            dd = (pb - pa) / L
            t, i = 0.0, 0
            while t < L:
                seg = pattern[i % len(pattern)] * mm
                if i % 2 == 0:
                    t2 = min(L, t + seg)
                    d.line([tuple(pa + dd * t), tuple(pa + dd * t2)], fill=colr, width=lw)
                t += seg
                i += 1
        for b in page:
            part = pm.parts[b["part"]]
            pc = b["piece"]
            F = part["faces"]
            tree = {e: (pk, ck) for pk, ck, e in pc["tree"]}
            tab_edges = {}
            for t, num in pc["tabs"]:
                tab_edges[(round(t[0][0], 9), round(t[0][1], 9), round(t[1][0], 9), round(t[1][1], 9))] = num
            for f, poly in pc["faces"].items():
                q2 = to_page(b, poly, lay["scale"]) * mm
                ff = F[f]
                for i in range(len(ff)):
                    e = tuple(sorted((ff[i], ff[(i + 1) % len(ff)])))
                    pa, pb = q2[i], q2[(i + 1) % len(ff)]
                    if e in tree:
                        pk, ck = tree[e]
                        if f != ck:
                            continue                                    # draw each fold once
                        n1, n2 = pc["normals"][pk], pc["normals"][ck]
                        ang = math.degrees(math.acos(max(-1.0, min(1.0, n1 @ n2))))
                        if ang < FLAT_DEG:
                            continue
                        c1 = part["verts"][F[pk]].mean(0)
                        c2 = part["verts"][F[ck]].mean(0)
                        mountain = (c2 - c1) @ n1 < 0
                        dashed(pa, pb, (1.6, 1.0) if mountain else (2.2, 0.7, 0.4, 0.7), (90, 90, 90))
                    else:
                        d.line([tuple(pa), tuple(pb)], fill=(0, 0, 0), width=lw)
            for t, num in pc["tabs"]:
                p = to_page(b, t, lay["scale"]) * mm
                dashed(p[0], p[1], (1.6, 1.0), (90, 90, 90))
                d.line([tuple(p[1]), tuple(p[2]), tuple(p[3]), tuple(p[0])], fill=(0, 0, 0), width=lw)
                put(img, tuple(p.mean(0)), str(num), f_small, (70, 70, 70), "mm")
            for f, e, num in pc["marks"]:
                poly = pc["faces"][f]
                ff = F[f]
                q2 = to_page(b, poly, lay["scale"]) * mm
                pa, pb = q2[ff.index(e[0])], q2[ff.index(e[1])]
                mid = (pa + pb) / 2
                inward = q2.mean(0) - mid
                L = np.linalg.norm(inward)
                if L < 1e-6:
                    continue
                c = mid + inward / L * min(2.4 * mm, L * 0.6)
                d.rounded_rectangle([c[0] - 2.2 * mm, c[1] - 1.4 * mm, c[0] + 2.2 * mm, c[1] + 1.4 * mm],
                                    radius=0.6 * mm, fill=(255, 255, 255))
                put(img, tuple(c), str(num), f_small, (0, 0, 0), "mm")
            # the part's name, on its biggest face
            big = max(pc["faces"], key=lambda k: abs(poly_area(pc["faces"][k])))
            c = to_page(b, pc["faces"][big], lay["scale"]).mean(0) * mm
            label = part["name"]
            tb = text_box(tuple(c), label, f_small, "mm")
            d.rectangle([tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2], fill=(255, 255, 255))
            put(img, tuple(c), label, f_small, (0, 0, 0), "mm")
        put(img, (MARGIN * mm, MARGIN * mm), title, f_big, (0, 0, 0))
        put(img, (A4[0] * mm - MARGIN * mm, MARGIN * mm + 1.0 * mm), "%d / %d" % (pn + 1, len(lay["pages"])),
            f_mid, (80, 80, 80), "ra")
        for i, line in enumerate(note if isinstance(note, (list, tuple)) else [note]):
            if line:
                put(img, (MARGIN * mm, (MARGIN + 5.6 + i * 3.2) * mm), line, f_small, (90, 90, 90))
        out.append(img)
    return out


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(x @ np.roll(y, -1) - y @ np.roll(x, -1))


def paint_face(a, q2, P3, pm, part_k, kind, bleed, retro):
    """Fill a face on the page: each pixel -> the point on the 3D face -> the part's colour map."""
    H, W, _ = a.shape
    x0, y0 = np.floor(q2.min(0) - bleed).astype(int) - 1
    x1, y1 = np.ceil(q2.max(0) + bleed).astype(int) + 2
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    p = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1)
    n = len(q2)
    cen = q2.mean(0)
    inside = np.ones(len(p), bool)
    if n == 3 or True:
        # star-shaped about the centre: inside every fan triangle's outer edge
        ins = np.zeros(len(p), bool)
        for i in range(n):
            tri = np.array([cen, q2[i], q2[(i + 1) % n]])
            ins |= in_triangle(p, tri, bleed)
        inside = ins
    p = p[inside]
    if not len(p):
        return
    # affine page -> 3D from three corners (well spread)
    i1 = n // 3
    i2 = 2 * n // 3
    A2 = np.c_[q2[[0, i1, i2]], np.ones(3)]
    T = np.linalg.lstsq(A2, P3[[0, i1, i2]], rcond=None)[0]
    pts3 = np.c_[p, np.ones(len(p))] @ T
    col = pm.colour(part_k, pts3, kind)
    if retro is not None:
        col = retro(col)
    px, py = p[:, 0].astype(int), p[:, 1].astype(int)
    a[py, px] = col


def in_triangle(p, tri, grow):
    """Points inside a triangle grown by `grow` pixels on its outer edge (tri[1]-tri[2]) only."""
    a, b, c = tri
    def side(u, v, w):
        e = v - u
        L = math.hypot(*e) or 1.0
        s = np.sign(e[0] * (w[1] - u[1]) - e[1] * (w[0] - u[0]))
        return (e[0] * (p[:, 1] - u[1]) - e[1] * (p[:, 0] - u[0])) * s / L
    return (side(a, b, c) >= -0.01) & (side(c, a, b) >= -0.01) & (side(b, c, a) >= -grow - 0.01)


def write_pdf(pages, dpi=200, quality=90):
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", resolution=dpi, save_all=True, append_images=pages[1:], quality=quality)
    return buf.getvalue()
