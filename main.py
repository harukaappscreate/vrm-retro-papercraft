"""VRM RETRO PAPERCRAFT - drop a VRM (0.x or 1.0) and get a papercraft of it: a PDF of A4 pages to print, cut,
fold and glue into a standing paper avatar (about 12-22 cm tall).

    python main.py                 then drop a .vrm file on the window
    python main.py avatar.vrm      open a file directly

The screen shows the paper pieces folding themselves up into the figure.
Mouse: drag to turn, wheel to zoom.  Keys: SPACE pause, F fold / unfold, R retro colours on / off,
UP / DOWN size, S save the PDF, P screenshot, G GIF.
In the browser, index.html adds a tool bar and a panel and talks to this app (see Bridge).

Nothing is uploaded anywhere: the file is read inside your own computer / browser.
Made with Pyxel 3 by Takashi Kitao (MIT License), NumPy (BSD-3-Clause), Pillow (MIT-CMU); in the browser, Pyodide (MPL-2.0).
"""
# SPDX-License-Identifier: MIT
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
import pyxel
from pyxel.cube import Camera, Mat4, Mesh, Node, Primitive, Quat, Vec3

import paper_layout as PL
import paper_model as PM
import vrm_io as V

W, H = 320, 240
UI = [0x000000, 0x2B335F, 0x7E2072, 0x19959C, 0x8B4852, 0x395C98, 0xA9C1FF, 0xEEEEEE,
      0xD4186C, 0xD38441, 0xE9C35B, 0x70C6A9, 0x7696DE, 0xA3A3A3, 0xFF9798, 0xEDC7B0]
BG = 1
INBOX = "/tmp/inbox"
TMP = tempfile.mkdtemp(prefix="vpc_")
HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [150, 180, 220, 260]                      # figure height, mm
CELL = 20                                          # one face in the preview atlas
# the loop of the preview, in frames
FLAT, FOLD, HOLD, OPEN = 60, 120, 150, 60
LOOP = FLAT + FOLD + HOLD + OPEN
try:
    import js                                      # in the browser (Pyodide) only
except ImportError:
    js = None


def ease(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def license_check(data):
    """(allowed, personal_only, reason).  Paper is a modified copy: no modification -> no papercraft."""
    j = V.read_glb(data)[0]
    ext = j.get("extensions", {})
    if "VRMC_vrm" in ext:
        m = ext["VRMC_vrm"].get("meta", {})
        mod = m.get("modification", "prohibited")
        return mod != "prohibited", not m.get("allowRedistribution", False), "modification: %s" % mod
    if "VRM" in ext:
        lic = ext["VRM"].get("meta", {}).get("licenseName", "")
        nd = lic.endswith("_ND")
        return not nd, lic in ("Redistribution_Prohibited", "") or "_NC" in lic, "licenseName: %s" % (lic or "-")
    return True, True, "no VRM license found"


def kabsch(A, B):
    """Rotation R and translation t with R @ a + t ~ b for the rows of A and B (no mirror)."""
    ca, cb = A.mean(0), B.mean(0)
    Hm = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def rot(axis, ang):
    x, y, z = axis
    c, s = math.cos(ang), math.sin(ang)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * k + (1 - c) * k @ k


def quat_from_matrix(R):
    q = V.mat_to_quat(R)
    return Quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


# ====================================================================== the page (browser) <-> this app
class Bridge:
    """index.html appends commands to pyxel._vpc (strings / JSON); this app reports back through js.vpc*()."""

    @staticmethod
    def commands():
        cmds = list(getattr(pyxel, "_vpc", None) or [])
        if cmds:
            pyxel._vpc = []
        return cmds

    @staticmethod
    def send(fn, data):
        if js is None:
            return
        try:
            getattr(js, fn)(json.dumps(data))
        except Exception:
            pass


# ====================================================================== the 3D preview: pieces that fold
class Preview:
    """Every paper piece as a tree of face nodes hinged on their parents (as in ONE GLB PAPER EARTH).
    Flat: the pieces lie on the table; folded: they stand together as the figure."""

    def __init__(self, pm, lay, lut):
        self.root = Node()
        self.sheets = Node()
        self.root.add_child(self.sheets)
        faces_all = []
        for b_i, (k, pc) in enumerate(lay["pieces"]):
            faces_all += [(k, f) for f in pc["faces"]]
        n = len(faces_all)
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = (n + cols - 1) // cols
        aw, ah = cols * CELL, rows * CELL
        self.img = pyxel.Image(aw, ah)
        A = np.ctypeslib.as_array(self.img.data_ptr()).reshape(ah, aw)
        A[:] = 7
        cell_of = {}
        for i, key in enumerate(faces_all):
            cell_of[key] = ((i % cols) * CELL, (i // cols) * CELL)
        self.pieces = []
        # the table: pieces in a grid, in model units, printed side up
        boxes = []
        for k, pc in lay["pieces"]:
            pts = np.concatenate(list(pc["faces"].values()))
            boxes.append((pts.min(0), pts.max(0)))
        span = max(max(hi[0] - lo[0], hi[1] - lo[1]) for lo, hi in boxes)
        gcols = max(1, int(math.ceil(math.sqrt(len(boxes)))))
        gap = span * 1.05
        centre = np.zeros(3)
        self.extent = gap * max(gcols, (len(boxes) + gcols - 1) // gcols)
        for i, ((k, pc), (lo, hi)) in enumerate(zip(lay["pieces"], boxes)):
            part = pm.parts[k]
            F, Vt = part["faces"], part["verts"]
            gx, gz = i % gcols, i // gcols
            off = np.array([(gx - (gcols - 1) / 2) * gap, 0.0, (gz - (len(boxes) / gcols - 1) / 2) * gap * 0.8])
            # the faces in tree order
            kids = {}
            for pk, ck, e in pc["tree"]:
                kids.setdefault(pk, []).append((ck, e))
            child = {ck for _, ck, _ in pc["tree"]}
            root_f = next(f for f in pc["faces"] if f not in child)
            nodes, hinge = {}, {}
            order = [root_f]
            for f in order:
                for ck, e in kids.get(f, []):
                    order.append(ck)
            par = {ck: (pk, e) for pk, ck, e in pc["tree"]}
            for f in order:
                corners = Vt[F[f]]
                # this face's picture in the atlas
                cx, cy = cell_of[(k, f)]
                loc2, _ = PL.flatten(corners)
                lo2, hi2 = loc2.min(0), loc2.max(0)
                sc = (CELL - 2) / max(1e-9, (hi2 - lo2).max())
                uv = (loc2 - lo2) * sc + 1
                gy, gx_ = np.mgrid[0:CELL, 0:CELL]
                pix = np.stack([gx_.ravel() + 0.5, gy.ravel() + 0.5], 1)
                inside = np.zeros(len(pix), bool)
                cen = uv.mean(0)
                for t in range(len(uv)):
                    inside |= PL.in_triangle(pix, np.array([cen, uv[t], uv[(t + 1) % len(uv)]]), 1.0)
                local = (pix - 1) / sc + lo2
                # local 2D -> 3D: affine from three corners
                i1, i2 = len(uv) // 3, 2 * len(uv) // 3
                T = np.linalg.lstsq(np.c_[loc2[[0, i1, i2]], np.ones(3)], corners[[0, i1, i2]], rcond=None)[0]
                p3 = np.c_[local, np.ones(len(local))] @ T
                col = pm.colour(k, p3, part["kind"][f])
                idx = lut.index(np.c_[col, np.full(len(col), 255, np.uint8)])
                blk = A[cy:cy + CELL, cx:cx + CELL].reshape(-1)
                blk[inside] = idx[inside]
                A[cy:cy + CELL, cx:cx + CELL] = blk.reshape(CELL, CELL)
                uvs = [c for (u, v) in uv for c in ((cx + u) / aw, (cy + v) / ah)]
                pos = [float(c) for c in corners.reshape(-1)]
                fan = []
                for t in range(1, len(corners) - 1):
                    fan += [0, t, t + 1]
                front = Primitive(Primitive.MODE_TRIANGLES, pos, fan, uvs=uvs, cull=Primitive.CULL_BACK)
                back = Primitive(Primitive.MODE_TRIANGLES, pos, fan[::-1], cull=Primitive.CULL_BACK)
                nd = Node()
                nd.add_child(Node.from_mesh(Mesh([front], [Mat4.IDENTITY], [-1], col_img=self.img)))
                nd.add_child(Node.from_mesh(Mesh([back], [Mat4.IDENTITY], [-1], col_img=7)))
                nodes[f] = nd
                if f in par:
                    pk, e = par[f]
                    nodes[pk].add_child(nd)
                    a, b = Vt[e[0]], Vt[e[1]]
                    axis = (b - a) / np.linalg.norm(b - a)
                    n1 = PL.flatten(Vt[F[pk]])[1]
                    n2 = PL.flatten(corners)[1]
                    ang = math.acos(max(-1.0, min(1.0, n1 @ n2)))
                    if np.linalg.norm(rot(axis, ang) @ n2 - n1) > np.linalg.norm(rot(axis, -ang) @ n2 - n1):
                        ang = -ang
                    hinge[f] = (a, axis, ang)
            # the flat pose of the root face: on the table, printed side up
            q2 = pc["faces"][root_f]
            n_out = PL.flatten(Vt[F[root_f]])[1]
            for sgn in (-1.0, 1.0):                                   # a flat face fits both ways: printed side up
                target = np.c_[q2[:, 0], np.zeros(len(q2)), sgn * q2[:, 1]] + off - centre
                Rr, tr = kabsch(Vt[F[root_f]], target)
                if (Rr @ n_out)[1] > 0:
                    break
            # a sheet of paper under the flat piece (hidden once it starts to fold)
            mg = span * 0.04
            x0, x1 = lo[0] - mg, hi[0] + mg
            z0, z1 = sorted((sgn * (lo[1] - mg), sgn * (hi[1] + mg)))
            sp = [x0, -0.02, z0, x1, -0.02, z0, x1, -0.02, z1, x0, -0.02, z1]
            sp = [c + o for c, o in zip(sp, list(off - centre) * 4)]
            sheet = Primitive(Primitive.MODE_TRIANGLES, sp, [0, 1, 2, 0, 2, 3], cull=Primitive.CULL_NONE)
            self.sheets.add_child(Node.from_mesh(Mesh([sheet], [Mat4.IDENTITY], [-1], col_img=13)))
            top = nodes[root_f]
            self.root.add_child(top)
            depth = {root_f: 0}
            for f in order[1:]:
                depth[f] = depth[par[f][0]] + 1
            self.pieces.append(dict(top=top, nodes=nodes, hinge=hinge, order=order, depth=depth,
                                    flat_q=quat_from_matrix(Rr), flat_t=tr, maxd=max(depth.values())))

    def pose(self, s):
        """s = 1 flat on the table, 0 folded up into the figure."""
        self.sheets.visible = s > 0.97
        for p in self.pieces:
            for f in p["order"][1:]:
                a, axis, ang = p["hinge"][f]
                lag = 0.4 * (p["maxd"] - p["depth"][f]) / max(1, p["maxd"])
                open_ = 1.0 - ease((1.0 - s) / 0.6 - lag / 0.6) if s < 1 else 1.0
                p["nodes"][f].transform = (Mat4.from_translation(Vec3(*a)) *
                                           Mat4.from_axis_angle(Vec3(*axis), math.degrees(ang * open_)) *
                                           Mat4.from_translation(Vec3(*(-a))))
            w = ease((s - 0.0) / 1.0)
            q = Quat.slerp(Quat.IDENTITY, p["flat_q"], w)
            t = p["flat_t"] * w
            p["top"].transform = Mat4.from_translation(Vec3(*t)) * Mat4.from_quat(q)


# ====================================================================== app
class App:
    def __init__(self, path=None):
        pyxel.init(W, H, title="VRM RETRO PAPERCRAFT", fps=30, capture_sec=8)
        pyxel.mouse(True)
        pyxel.colors[:] = UI + [0x808080] * 240
        self.stage = Node()
        cam = Camera()
        cam.fov = 30.0
        cam.near = 0.05
        cam.far = 50.0
        cam.clear_color = BG
        self.stage.camera = cam
        self.cam = cam
        self.world = Node()
        self.stage.add_child(self.world)
        self.state, self.msg, self.err = "drop", "", ""
        self.pending = path
        self.yaw, self.pitch, self.dist = 20.0, 14.0, 4.3
        self.drag = None
        self.size = 1                                # SIZES index
        self.retro = False
        self.frame, self.paused, self.fold_only = 0, False, None
        self.model = self.pm = self.pm_retro = self.lay = self.preview = None
        self.allowed, self.personal = True, True
        self.name = ""
        self.job = None
        os.makedirs(INBOX, exist_ok=True)
        self.seen = set(os.listdir(INBOX))
        self.used = set()
        pyxel.run(self.update, self.draw)

    # ------------------------------------------------------------------ files
    def new_file(self):
        for f in list(getattr(pyxel, "dropped_files", None) or []) + list(getattr(pyxel, "_dropped_files", None) or []):
            if f not in self.used:
                self.used.add(f)
                return f
        try:
            for f in sorted(os.listdir(INBOX)):
                if f not in self.seen:
                    self.seen.add(f)
                    return os.path.join(INBOX, f)
        except OSError:
            pass
        return None

    def say(self, text, error=False):
        self.msg = text.upper()
        Bridge.send("vpcStatus", {"text": text, "error": error})

    def load(self, path):
        name = os.path.basename(path)
        self.name = name.split("_", 1)[-1] if path.startswith(INBOX) else name
        self.say("Reading %s ..." % self.name)
        self.state = "busy"
        self.job = ("load", path)

    def do_load(self, path):
        try:
            data = open(path, "rb").read()
            model = V.VrmModel(data, tex_size=512)
            allowed, personal, reason = license_check(data)
        except Exception as e:
            self.state = "drop" if self.pm is None else "view"
            self.say("Cannot read this file: %s" % e, True)
            return
        finally:
            try:
                if path.startswith(INBOX):
                    os.remove(path)
            except OSError:
                pass
        self.model, self.allowed, self.personal = model, allowed, personal
        info = dict(title=model.title, author=model.author, version=model.version, license=model.license,
                    warnings=model.warnings, allowed=allowed, personal=personal, tris=model.tris)
        Bridge.send("vpcInfo", info)
        if not allowed:
            self.state = "refused"
            self.pm = self.lay = self.preview = None
            self.world_clear()
            self.say("Modification is not allowed for this avatar (%s): no papercraft." % reason, True)
            return
        self.say("Making the paper figure ...")
        self.job = ("build", None)

    def do_build(self):
        t0 = time.perf_counter()
        try:
            pal = self.model.palette
            pyxel.colors[16:16 + len(pal)] = [int(r) << 16 | int(g) << 8 | int(b) for r, g, b in pal]
            self.lut = V.Lut(pal, 16)
            self.pm = PM.PaperModel(self.model, SIZES[self.size])
            self.pm_retro = PM.retro_copy(self.pm)
            self.make_layout()
        except Exception as e:
            self.state = "drop" if self.preview is None else "view"
            self.say("Could not make the papercraft: %s" % e, True)
            return
        self.build_ms = (time.perf_counter() - t0) * 1000

    def make_layout(self):
        pm = self.pm_retro if self.retro else self.pm
        try:
            self.lay = PL.layout(self.pm, SIZES[self.size])
        except ValueError as e:
            self.say(str(e), True)
            return
        self.lay["pieces"] = [(b["part"], b["piece"]) for page in self.lay["pages"] for b in page]
        self.world_clear()
        self.preview = Preview(pm, self.lay, self.lut)
        self.world.add_child(self.preview.root)
        self.frame = 0
        self.state = "view"
        edges = []
        for part in self.pm.parts:
            for f in part["faces"]:
                c = part["verts"][f]
                edges += list(np.linalg.norm(c - np.roll(c, -1, 0), axis=1))
        self.shortest = min(edges) * self.lay["scale"]
        info = dict(pages=len(self.lay["pages"]), pieces=self.lay["n_pieces"], tabs=self.lay["n_tabs"],
                    faces=sum(len(p["faces"]) for p in self.pm.parts), height=SIZES[self.size],
                    shortest=round(self.shortest, 1), retro=self.retro)
        Bridge.send("vpcMade", info)
        self.say("%d pages, %d pieces. Press Download PDF." % (info["pages"], info["pieces"]))

    def world_clear(self):
        for ch in list(self.world.children):
            self.world.remove_child(ch)

    def pdf(self):
        if self.lay is None:
            return
        pm = self.pm_retro if self.retro else self.pm
        notes = ["%s by %s. Print at 100%% (actual size). Cut the solid lines; fold dashed = mountain, "
                 "dash-dot = valley; glue tab N under edge N." % (self.model.title[:40], self.model.author[:30]),
                 ("For your own printing only: do not share this PDF or the finished figure."
                  if self.personal else "Follow the license of the avatar when you share the finished figure.")]
        t0 = time.perf_counter()
        pages = PL.render_pages(pm, self.lay, dpi=200, note=notes)
        data = PL.write_pdf(pages, dpi=200)
        base = os.path.splitext(self.name)[0] or "avatar"
        path = os.path.join(TMP, "%s_papercraft_%dmm.pdf" % (base[:40], SIZES[self.size]))
        with open(path, "wb") as f:
            f.write(data)
        self.deliver(path)
        self.say("Saved %s (%d pages, %.1f MB, %.1f s)." % (os.path.basename(path), len(pages), len(data) / 1e6,
                                                            time.perf_counter() - t0))

    def deliver(self, path):
        """Browser: start a download.  Desktop: copy next to main.py."""
        if js is not None:
            js._savePyxelFile(path)
        else:
            import shutil
            shutil.copy(path, os.path.join(HERE, os.path.basename(path)))

    # ------------------------------------------------------------------ update / draw
    def update(self):
        if self.pending:
            self.load(self.pending)
            self.pending = None
        if self.job and self.state == "busy" and self.frame_busy():
            kind, arg = self.job
            self.job = None
            if kind == "load":
                self.do_load(arg)
            elif kind == "build":
                self.do_build()
            elif kind == "layout":
                self.make_layout()
            elif kind == "pdf":
                self.pdf()
                self.state = "view"
        f = self.new_file()
        if f and self.state != "busy":
            self.load(f)
        for c in Bridge.commands():
            self.command(c)
        if pyxel.btnp(pyxel.KEY_S):
            self.command("pdf")
        if pyxel.btnp(pyxel.KEY_R):
            self.command("retro:%d" % (0 if self.retro else 1))
        if pyxel.btnp(pyxel.KEY_UP):
            self.command("size:%d" % SIZES[min(len(SIZES) - 1, self.size + 1)])
        if pyxel.btnp(pyxel.KEY_DOWN):
            self.command("size:%d" % SIZES[max(0, self.size - 1)])
        if pyxel.btnp(pyxel.KEY_SPACE):
            self.paused = not self.paused
        if pyxel.btnp(pyxel.KEY_F):
            self.command("fold")
        if pyxel.btnp(pyxel.KEY_P):
            pyxel.screenshot()
        if pyxel.btnp(pyxel.KEY_G):
            pyxel.screencast()
        # the camera
        if pyxel.btn(pyxel.MOUSE_BUTTON_LEFT):
            if self.drag:
                self.yaw -= (pyxel.mouse_x - self.drag[0]) * 0.8
                self.pitch = max(-10.0, min(80.0, self.pitch + (pyxel.mouse_y - self.drag[1]) * 0.6))
            self.drag = (pyxel.mouse_x, pyxel.mouse_y)
        else:
            self.drag = None
        if pyxel.mouse_wheel:
            self.dist = max(1.2, min(8.0, self.dist * (0.9 ** pyxel.mouse_wheel)))
        if self.preview is not None:
            if not self.paused:
                self.frame += 1
            s = self.fold_value()
            self.preview.pose(s)
            # flat: look down on the table; folded: look at the figure
            w = ease(1 - s)
            h = self.pm.height
            target = Vec3(0, self.pm.floor + h * (0.5 * w) + 0.0, 0)
            pitch = self.pitch * w + 70 * (1 - w)
            dist = self.dist * (h / 1.6) * w + self.preview.extent * 2.1 * (1 - w)
            yaw = self.yaw + (self.frame * 0.4 if w > 0.99 and not self.paused else 0)
            yr, pr = math.radians(yaw), math.radians(pitch)
            eye = Vec3(target.x + dist * math.cos(pr) * math.sin(yr), target.y + dist * math.sin(pr),
                       target.z + dist * math.cos(pr) * math.cos(yr))
            self.cam.transform = Mat4.look_at(eye, target)
        self.stage.update()

    def frame_busy(self):
        """Busy jobs run one frame after their message is drawn."""
        self.busy_wait = getattr(self, "busy_wait", 0) + 1
        if self.busy_wait >= 2:
            self.busy_wait = 0
            return True
        return False

    def fold_value(self):
        if self.fold_only is not None:
            return self.fold_only
        f = self.frame % LOOP
        if f < FLAT:
            return 1.0
        f -= FLAT
        if f < FOLD:
            return 1.0 - f / FOLD
        f -= FOLD
        if f < HOLD:
            return 0.0
        return (f - HOLD) / OPEN

    def command(self, c):
        if c == "pdf":
            if self.lay is not None and self.state == "view":
                self.state = "busy"
                self.say("Drawing the pages ...")
                self.job = ("pdf", None)
        elif c.startswith("size:"):
            v = int(c.split(":")[1])
            if v in SIZES and self.pm is not None:
                self.size = SIZES.index(v)
                self.state = "busy"
                self.say("Making the paper figure for %d mm ..." % v)
                self.job = ("build", None)
            elif v in SIZES:
                self.size = SIZES.index(v)
        elif c.startswith("retro:"):
            self.retro = c.endswith("1")
            if self.pm is not None:
                self.state = "busy"
                self.say("Colours: %s ..." % ("retro" if self.retro else "full"))
                self.job = ("layout", None)
        elif c == "fold":
            self.fold_only = None if self.fold_only is not None else (0.0 if self.fold_value() > 0.5 else 1.0)
        elif c.startswith("loop"):
            self.fold_only = None
        elif c == "pause":
            self.paused = not self.paused
        elif c == "view:reset":
            self.yaw, self.pitch, self.dist = 20.0, 14.0, 4.3
        elif c == "shot":
            pyxel.screenshot()
        elif c == "gif":
            pyxel.screencast()

    def draw(self):
        pyxel.cls(BG)
        if self.preview is not None and self.state in ("view", "busy"):
            self.stage.draw(0, 0, W, H)
        if self.state == "drop" or (self.state == "refused"):
            pyxel.text(W // 2 - 66, H // 2 - 12, "DROP A VRM (0.X / 1.0) HERE", 7)
            pyxel.text(W // 2 - 76, H // 2, "IT BECOMES A PAPERCRAFT TO PRINT", 6)
        pyxel.text(6, 6, "VRM RETRO PAPERCRAFT", 10)
        if self.preview is not None:
            pyxel.text(6, 14, "%d MM  %s  %d PAGES" % (SIZES[self.size], "RETRO" if self.retro else "FULL COLOUR",
                                                       len(self.lay["pages"])), 6)
        if self.msg:
            col = 8 if self.state == "refused" else 7
            for i, line in enumerate(wrap(self.msg, 76)[:3]):
                pyxel.text(6, H - 26 + i * 8, line, col)
        if js is None:
            pyxel.text(6, H - 34, "S PDF  R RETRO  UP/DOWN SIZE  F FOLD  SPACE PAUSE", 13)


def wrap(text, n):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


if __name__ == "__main__":
    App(sys.argv[1] if len(sys.argv) > 1 else None)
