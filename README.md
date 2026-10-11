# Curvature animations for the Three Gap Theorem talk

Three short Manim scenes supporting the slide on Theorem 1 of Biringer & Schmidt,
*The Three Gap Theorem and Riemannian geometry*, Geom. Dedicata **136** (2008) 175–190.

All the geometry is computed, not drawn by hand: the spherical triangle is a genuine
great-circle triangle in orthographic projection, the hyperbolic one is a genuine
Poincaré-disc geodesic triangle, and the rim counts in `RoomInABall` come from the
comparison formulas. `kappa_animation.py` asserts the counts quoted in the talk on
import, so the scenes cannot silently drift from the slide.

## Rendering

```sh
uv sync

# check layout fast
uv run manim -ql kappa_animation.py ThreeGeometries Spreading RoomInABall

# final 1080p60 MP4s            -> media/videos/kappa_animation/1080p60/
uv run manim -qh kappa_animation.py ThreeGeometries Spreading RoomInABall

# final-frame PNGs              -> media/images/kappa_animation/
uv run manim -qh -s kappa_animation.py ThreeGeometries Spreading RoomInABall

# 15 fps PNG sequences          -> frames/images/kappa_animation/<Scene>/
uv run manim -qh --fps 15 --format=png-sequence -0 4 --media_dir frames \
  kappa_animation.py ThreeGeometries Spreading RoomInABall
```

The count table behind `RoomInABall`:

```sh
uv run python kappa_animation.py
```

## Outputs

| Scene | Duration | MP4 | Final-frame PNG | PNG sequence |
|---|---|---|---|---|
| `ThreeGeometries` | 12.4 s | `media/videos/kappa_animation/1080p60/ThreeGeometries.mp4` | `media/images/kappa_animation/ThreeGeometries_ManimCE_v0.22.0.png` | `frames/images/kappa_animation/ThreeGeometries/0000.png … 0115.png` |
| `Spreading` | 12.1 s | `media/videos/kappa_animation/1080p60/Spreading.mp4` | `media/images/kappa_animation/Spreading_ManimCE_v0.22.0.png` | `frames/images/kappa_animation/Spreading/0000.png … 0114.png` |
| `RoomInABall` | 15.5 s | `media/videos/kappa_animation/1080p60/RoomInABall.mp4` | `media/images/kappa_animation/RoomInABall_ManimCE_v0.22.0.png` | `frames/images/kappa_animation/RoomInABall/0000.png … 0182.png` |

All 1920×1080. MP4s are 60 fps; the PNG sequences are 15 fps and omit the static
holds at the end of each scene, so they are a little shorter than the MP4s.

## Embedding in Beamer

Video (needs a PDF viewer that plays media, e.g. Okular or Acrobat):

```latex
\usepackage{multimedia}
\movie[width=\textwidth,height=0.5625\textwidth,autostart]
  {\includegraphics[width=\textwidth]{media/images/kappa_animation/ThreeGeometries_ManimCE_v0.22.0.png}}
  {media/videos/kappa_animation/1080p60/ThreeGeometries.mp4}
```

Self-contained frame sequence (plays in Acrobat; `\animategraphics` needs
`pdflatex`/`lualatex`, not `dvips`):

```latex
\usepackage{animate}
\animategraphics[width=\textwidth,autoplay,controls]{15}%
  {frames/images/kappa_animation/ThreeGeometries/}{0000}{0115}
```

Static fallback — just the final-frame PNG:

```latex
\includegraphics[width=\textwidth]{media/images/kappa_animation/ThreeGeometries_ManimCE_v0.22.0.png}
```

## The mathematics

`ThreeGeometries` — the three model spaces `M_κ` and the sign of the angle-sum defect.

`Spreading` — two unit-speed geodesics leaving a point at angle θ = 60°, and the
distance `d(t)` between their time-`t` points:

- flat: `d = 2t sin(θ/2)`
- sphere, κ = 1: `cos d = cos²t + sin²t cos θ`
- hyperbolic, κ = −1: `cosh d = cosh²t − sinh²t cos θ`

Positive curvature pulls the geodesics back together (on the unit sphere they meet
again at `t = π`); negative curvature drives them apart exponentially. This is the
picture behind Toponogov comparison: `sec ≥ κ` means geodesics in `M` spread no
faster than in `M_κ`.

`RoomInABall` — two points at distance `r` from `p`, an angle φ apart, are themselves
`≥ r` apart exactly when `cos φ ≤ c(r)`, with

- flat: `c = 1/2`
- sphere, κ = 1: `c = cos r / (1 + cos r)`
- hyperbolic, κ = −1: `c = cosh r / (cosh r + 1)`

The displayed count is the largest `N` with `360°/N > arccos c(r)`, strict because the
ball is open. It is labelled *points around the edge*, not the packing number `P(M, r)`:
for `κ < 0` and large `r`, further points fit at other radii, so this is a lower bound
on `P`. As `r` grows the flat count stays at 5, the spherical count falls, and the
hyperbolic count climbs — which is why Theorem 1 needs `diam(M) ≤ D` only when `κ < 0`.
