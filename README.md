# VRM RETRO PAPERCRAFT

[日本語](README.ja.md)

Open a VRM avatar (0.x or 1.0) and get a **papercraft** of it: A4 pages (PDF) to print, cut, fold and glue into a standing paper figure, 150–260 mm tall.
The screen shows the paper pieces folding themselves up into the figure.

A sibling of [VRM RETRO VIEWER](https://github.com/harukaappscreate/vrm-retro-viewer), [VRM Motion Studio](https://github.com/harukaappscreate/vrm-motion-studio) and [VRM RETRO MAKER](https://github.com/harukaappscreate/vrm-retro-maker).

Your file is never uploaded. Reading and converting happen inside your browser (or your own computer).

## How it works

1. **Parts.** Every vertex goes to the paper part of its strongest bone: head (with the hair), body (with the shoulders and the skirt), two arms (hands as mittens), two legs, two feet.
2. **Paper solids ("loft").** Each part becomes a closed tube along its bone: rings of 6–8 corners at 3–6 levels, a flat cap at each end. The radius of each corner is read from the avatar's surface (96th percentile), so hair, clothes and shoes are inside. Thin arms and legs are made a little thicker, so that no edge is shorter than 5 mm.
3. **Colours.** About 240,000 points are spread over the avatar and coloured from its textures. Each part keeps a small map of the outermost colour (along the bone × around it, and one for each cap). Every pixel of the paper looks up the map where it sits on the 3D part, so the face, the clothes and the shoes print where they belong.
4. **Unfolding.** Each band of the tube unrolls as one strip, cut open at the back (head, body), underneath (arms) or inside (legs). The widest band is the spine; the others hang off it opposite the seam, the caps off the end bands. A face that would overlap starts a new piece.
5. **Glue tabs.** One tab for each cut edge, numbered: glue tab N under the edge marked N. Where there is no room for a tab, the number is printed on both edges: join them with a little tape from inside.
6. **Pages.** The pieces are turned to their lowest box and packed on A4 shelves (first fit, tallest first). Lines: solid = cut, dashed = mountain fold, dash-dot = valley fold. The colours bleed under the cut lines so no white edges show.

**Retro colours** use chunkier colour maps and one palette of 24 colours, like the rest of the RETRO family.

## The license of the avatar

A papercraft is a modified copy. Avatars whose license forbids modification are refused (VRM 1.0 `modification: prohibited`; VRM 0.x `*_ND`).
If the avatar may not be redistributed, the PDF says on every page: print for yourself only, do not share the PDF or the finished figure.
Only use avatars you made or may use. No characters that belong to someone else.

## Checks (sample avatar, 180 mm)

- 8 parts, 16 pieces, 400 faces, 244 cut edges, about 222 of them with a tab; 2 A4 pages.
- Every face is placed flat next to its neighbour across their shared edge, so glued edges have the same length; no overlaps inside a piece (separating-axis tests); shortest edge 5.0 mm.
- Outline of the paper figure against the avatar (8 directions): 0.74–0.78 overlap. Lost: the gaps between the legs and under the skirt, fingers, hair strands.
- Timing (desktop Python): reading 0.4 s, the paper figure 0.5 s, the layout 0.3 s, the PDF 1.3 s.
- Not yet: a figure printed and built.

## Run it yourself

You need Python 3 and Pyxel 3 (`pip install pyxel numpy pillow`).

```bash
python main.py                 # then drop a .vrm file on the window
python main.py avatar.vrm
```

Keys: S save the PDF (next to main.py), R retro colours on / off, UP / DOWN size, F fold / unfold, SPACE pause, P screenshot, G GIF. Mouse: drag to turn, wheel to zoom.

Build the itch.io upload: `python tools/build_web.py` (makes `vrm_retro_papercraft_itch.zip`: index.html, the .pyxapp, the sample avatar).

## Files

- `main.py`: the app (desktop and browser)
- `paper_model.py`: the avatar as paper solids, and their colour maps
- `paper_layout.py`: unfolding, glue tabs, pages, PDF
- `vrm_io.py`: reading VRM 0.x / 1.0 (the same as in VRM RETRO VIEWER / MAKER)
- `web/index.html`: the browser page; `web/sample.vrm`: the sample avatar

## Credits

- Made with [Pyxel 3](https://github.com/kitao/pyxel) by Takashi Kitao (MIT License). Runs on Pyodide (MPL-2.0), NumPy (BSD-3-Clause) and Pillow (MIT-CMU).
- Sample avatar: Sendagaya Shibu, a CC0 VRoid sample model (VRM data by coati; redistribution and modification allowed). For this tool its textures were reduced and its facial expressions removed.
- Tool by haruka_apps. MIT License.

Provided "AS IS", without warranty of any kind.
