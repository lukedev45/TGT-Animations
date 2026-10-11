"""
kappa_animation.py -- three short Manim scenes explaining the curvature bound kappa
for a talk on Biringer & Schmidt, "The Three Gap Theorem and Riemannian geometry".

Scenes
  ThreeGeometries  -- the three model spaces M_kappa and their geodesic triangles
  Spreading        -- how fast two geodesics at angle theta separate in each model
  RoomInABall      -- how many points, pairwise >= r apart, fit around the rim of an r-ball

Render (from the project root, which has manim in .venv):

  # low quality, for checking layout
  uv run manim -ql kappa_animation.py ThreeGeometries Spreading RoomInABall

  # final 1080p MP4s
  uv run manim -qh kappa_animation.py ThreeGeometries Spreading RoomInABall

  # final-frame PNG (static fallback for the PDF)
  uv run manim -qh -s kappa_animation.py ThreeGeometries Spreading RoomInABall

  # PNG frame sequence at 15 fps, for the LaTeX `animate` package
  uv run manim -qh --format=png --fps 15 kappa_animation.py ThreeGeometries Spreading RoomInABall

Running this file directly prints the rim-count table and checks it against the
values quoted in the talk:  uv run python kappa_animation.py
"""

import math

from manim import *

# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------

config.background_color = WHITE   # blend with the Beamer slides
# resolution/fps come from the CLI quality flag: -ql to check layout, -qh for 1920x1080

PRIMARY = ManimColor("#005493")  # positive curvature / sphere
ACCENT = ManimColor("#BE3C28")   # negative curvature / hyperbolic
FLAT = ManimColor("#3C3C3C")     # zero curvature / Euclidean
GREY = ManimColor("#9AA0A6")     # construction lines
INK = ManimColor("#1A1A1A")      # body text

THETA = math.radians(60.0)       # the fixed angle used in Spreading


def mtex(s, **kwargs):
    kwargs.setdefault("color", INK)
    return MathTex(s, **kwargs)


def label(s, **kwargs):
    """Plain text, for live-updating numbers (Pango: no LaTeX run per frame)."""
    kwargs.setdefault("color", INK)
    return Text(s, **kwargs)


# ---------------------------------------------------------------------------
# the geometry, done numerically
# ---------------------------------------------------------------------------


def c_of_r(r, kappa):
    """Threshold c(r): two points at distance r from p, an angle phi apart, are
    themselves >= r apart exactly when cos(phi) <= c(r)."""
    if kappa > 0:                               # unit sphere, kappa = 1
        denom = 1.0 + math.cos(r)
        if denom <= 1e-12:
            return -math.inf
        return math.cos(r) / denom
    if kappa == 0:                              # Euclidean
        return 0.5
    return math.cosh(r) / (math.cosh(r) + 1.0)  # hyperbolic, kappa = -1


def min_angle(r, kappa):
    """Least angle at p between two such points, in radians; None if no two
    points at distance r from p are ever >= r apart."""
    c = c_of_r(r, kappa)
    if c <= -1.0:
        return None
    return math.acos(max(-1.0, min(1.0, c)))


def rim_count(r, kappa):
    """Largest N with 360/N > min_angle: how many points fit around the rim of
    the *open* ball of radius r, pairwise >= r apart. Strict, since the ball is open."""
    phi = min_angle(r, kappa)
    if phi is None or phi <= 1e-12:
        return 1
    n = int(math.floor(2.0 * math.pi / phi))
    while n >= 2 and 2.0 * math.pi / n <= phi + 1e-12:
        n -= 1
    return max(n, 1)


def spread(t, kappa, theta=THETA):
    """Distance between the time-t points of two unit-speed geodesics leaving a
    common point at angle theta, in the model space of curvature kappa."""
    if kappa > 0:
        c = math.cos(t) ** 2 + math.sin(t) ** 2 * math.cos(theta)
        return math.acos(max(-1.0, min(1.0, c)))
    if kappa == 0:
        return 2.0 * t * math.sin(theta / 2.0)
    return math.acosh(max(1.0, math.cosh(t) ** 2 - math.sinh(t) ** 2 * math.cos(theta)))


def _self_check():
    """The counts quoted on the slide. Checked on import so the scenes cannot
    silently disagree with the talk."""
    assert all(rim_count(r, 0) == 5 for r in (0.3, 0.5, 1.0, 1.7, 2.5, 7.0))
    assert rim_count(1.0, 1) == 5
    assert rim_count(1.5, 1) == 4
    assert rim_count(1.0, -1) == 6
    assert rim_count(2.0, -1) == 9


_self_check()


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------


def _curve(points, color, width=5.0):
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_smoothly(points)
    return m


def _sphere_triangle(radius, alpha_deg=48.0, tilt_deg=22.0, samples=48):
    """Orthographic projection of a genuine equilateral great-circle triangle on
    the unit sphere: vertices at polar angle alpha, 120 degrees apart, tilted up
    so the triangle faces the viewer. Angle sum = 3*arccos(cos a/(1+cos a)) > pi."""
    beta = math.radians(tilt_deg)
    rot = np.array([[1.0, 0.0, 0.0],
                    [0.0, math.cos(beta), -math.sin(beta)],
                    [0.0, math.sin(beta), math.cos(beta)]])
    a = math.radians(alpha_deg)
    verts3 = []
    for phi in (90.0, 210.0, 330.0):
        p = math.radians(phi)
        verts3.append(rot @ np.array([math.sin(a) * math.cos(p),
                                      math.sin(a) * math.sin(p),
                                      math.cos(a)]))

    def project(v):
        return np.array([v[0] * radius, v[1] * radius, 0.0])

    sides = []
    for i in range(3):
        u, v = verts3[i], verts3[(i + 1) % 3]
        omega = math.acos(max(-1.0, min(1.0, float(np.dot(u, v)))))
        pts = []
        for k in range(samples + 1):
            s = k / samples
            w = (math.sin((1.0 - s) * omega) * u + math.sin(s * omega) * v) / math.sin(omega)
            pts.append(project(w / np.linalg.norm(w)))
        sides.append(pts)
    return sides, [project(v) for v in verts3]


def _poincare_triangle(radius, rho=0.84, samples=48):
    """A genuine geodesic triangle in the Poincare disc: vertices at Euclidean
    radius rho, 120 degrees apart; sides are the circular arcs orthogonal to the
    ideal boundary. Angle sum < pi."""
    verts = []
    for phi in (90.0, 210.0, 330.0):
        p = math.radians(phi)
        verts.append(np.array([rho * math.cos(p), rho * math.sin(p)]))

    sides = []
    for i in range(3):
        z1, z2 = verts[i], verts[(i + 1) % 3]
        # centre w of the orthogonal circle: 2 <w, z_i> = |z_i|^2 + 1
        a = np.array([[2.0 * z1[0], 2.0 * z1[1]], [2.0 * z2[0], 2.0 * z2[1]]])
        b = np.array([z1 @ z1 + 1.0, z2 @ z2 + 1.0])
        w = np.linalg.solve(a, b)
        rad = math.sqrt(w @ w - 1.0)
        a1 = math.atan2(z1[1] - w[1], z1[0] - w[0])
        a2 = math.atan2(z2[1] - w[1], z2[0] - w[0])
        sweep = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi  # short way round
        pts = []
        for k in range(samples + 1):
            ang = a1 + sweep * k / samples
            pts.append(np.array([(w[0] + rad * math.cos(ang)) * radius,
                                 (w[1] + rad * math.sin(ang)) * radius, 0.0]))
        sides.append(pts)
    return sides, [np.array([v[0] * radius, v[1] * radius, 0.0]) for v in verts]


def _flat_triangle(radius, rho=0.84):
    verts = []
    for phi in (90.0, 210.0, 330.0):
        p = math.radians(phi)
        verts.append(np.array([rho * radius * math.cos(p), rho * radius * math.sin(p), 0.0]))
    sides = [[verts[i], verts[(i + 1) % 3]] for i in range(3)]
    return sides, verts


# ---------------------------------------------------------------------------
# Scene 1
# ---------------------------------------------------------------------------


class ThreeGeometries(Scene):
    """The three model spaces, with a geodesic triangle on each."""

    def construct(self):
        radius = 1.5
        centres = [LEFT * 4.75, ORIGIN, RIGHT * 4.75]
        colors = [PRIMARY, FLAT, ACCENT]
        kappas = [r"\kappa > 0", r"\kappa = 0", r"\kappa < 0"]
        names = ["sphere", "flat plane", "hyperbolic plane"]
        sums = [r"\alpha + \beta + \gamma > \pi",
                r"\alpha + \beta + \gamma = \pi",
                r"\alpha + \beta + \gamma < \pi"]

        backdrops, triangles, heads, subs, feet = (VGroup() for _ in range(5))

        for i, (c, col) in enumerate(zip(centres, colors)):
            shift = c + UP * 0.2

            if i == 0:
                back = VGroup(Circle(radius=radius, stroke_color=col, stroke_width=3,
                                     fill_color=col, fill_opacity=0.07))
                for frac, h in ((-0.45, 0.30), (0.45, 0.30)):
                    back.add(Ellipse(width=2 * radius * math.sqrt(1 - frac ** 2),
                                     height=2 * radius * h * math.sqrt(1 - frac ** 2),
                                     stroke_color=GREY, stroke_width=1.5,
                                     stroke_opacity=0.55).shift(UP * frac * radius))
                sides, _ = _sphere_triangle(radius)
            elif i == 1:
                back = VGroup()
                half = radius * 0.98
                for k in range(-3, 4):
                    u = half * k / 3.0
                    back.add(Line([-half, u, 0], [half, u, 0], stroke_color=GREY,
                                  stroke_width=1.5, stroke_opacity=0.55))
                    back.add(Line([u, -half, 0], [u, half, 0], stroke_color=GREY,
                                  stroke_width=1.5, stroke_opacity=0.55))
                sides, _ = _flat_triangle(radius)
            else:
                back = VGroup(
                    Circle(radius=radius, stroke_width=0, fill_color=col, fill_opacity=0.07),
                    DashedVMobject(Circle(radius=radius, stroke_color=col, stroke_width=3),
                                   num_dashes=64, dashed_ratio=0.55),
                )
                sides, _ = _poincare_triangle(radius)

            tri = VGroup(*[_curve(s, col, 5.5) if len(s) > 2
                           else Line(s[0], s[1], stroke_color=col, stroke_width=5.5)
                           for s in sides])
            # fill: trace the actual boundary as dense corners, so the straight
            # sides of the flat triangle stay straight
            face = VMobject(fill_color=col, fill_opacity=0.16, stroke_width=0)
            face.set_points_as_corners(
                [p for s in sides for p in s[:-1]] + [sides[0][0]])

            backdrops.add(back.shift(shift))
            triangles.add(VGroup(face, tri).shift(shift))
            heads.add(mtex(kappas[i], color=col, font_size=58).move_to(c + UP * 3.05))
            subs.add(label(names[i], font_size=28, color=GREY).move_to(c + UP * 2.4))
            feet.add(mtex(sums[i], color=col, font_size=42).move_to(c + DOWN * 2.3))

        hint = label("geodesic triangles", font_size=30, color=INK).move_to(DOWN * 3.3)

        self.play(LaggedStart(*[FadeIn(h, shift=DOWN * 0.3) for h in heads],
                              lag_ratio=0.18, run_time=1.1))
        self.play(LaggedStart(*[FadeIn(s) for s in subs], lag_ratio=0.18, run_time=0.7),
                  LaggedStart(*[FadeIn(b) for b in backdrops], lag_ratio=0.18, run_time=1.1))
        self.play(FadeIn(hint, run_time=0.6))
        self.play(LaggedStart(*[Create(t) for t in triangles], lag_ratio=0.25, run_time=3.2))
        self.play(LaggedStart(*[FadeIn(f, shift=UP * 0.25) for f in feet],
                              lag_ratio=0.22, run_time=1.6))
        self.wait(4.8)


# ---------------------------------------------------------------------------
# Scene 2
# ---------------------------------------------------------------------------


class Spreading(Scene):
    """How fast two geodesics at a fixed angle pull apart, in each model space."""

    def construct(self):
        t_max = math.pi

        # --- left: what d(t) means -------------------------------------------
        origin = np.array([-5.05, 0.55, 0.0])
        arm = 2.0
        ends = []
        arms = VGroup()
        for sgn in (+1, -1):
            ang = math.pi / 2 + sgn * THETA / 2
            e = origin + arm * np.array([math.cos(ang), math.sin(ang), 0.0])
            ends.append(e)
            arms.add(Line(origin, e, stroke_color=GREY, stroke_width=4))
        wedge = Arc(radius=0.52, start_angle=math.pi / 2 - THETA / 2, angle=THETA,
                    arc_center=origin, stroke_color=INK, stroke_width=3)
        theta_lab = mtex(r"\theta = 60^\circ", font_size=34).move_to(
            origin + RIGHT * 1.22 + UP * 0.30)
        chord = DashedLine(ends[1], ends[0], stroke_color=INK, stroke_width=4, dash_length=0.12)
        d_lab = mtex(r"d(t)", font_size=38).move_to((ends[0] + ends[1]) / 2 + UP * 0.42)
        p_dot = Dot(origin, radius=0.075, color=INK)
        p_lab = mtex(r"p", font_size=34).move_to(origin + DOWN * 0.38)
        tips = VGroup(*[Dot(e, radius=0.065, color=GREY) for e in ends])
        t_lab = mtex(r"t", font_size=34, color=GREY).move_to(
            origin + 0.60 * (ends[0] - origin) + LEFT * 0.32)  # ends[0] is the left arm

        schematic = VGroup(arms, wedge, theta_lab, chord, d_lab, p_dot, p_lab, tips, t_lab)

        # --- right: the three curves -----------------------------------------
        axes = Axes(
            x_range=[0, 3.35, 1], y_range=[0, 5.3, 1],
            x_length=7.1, y_length=5.5,
            axis_config={"color": INK, "stroke_width": 3,
                         "include_ticks": True, "tick_size": 0.07, "font_size": 26,
                         "decimal_number_config": {"num_decimal_places": 0}},
            x_axis_config={"numbers_to_include": [1, 2, 3]},
            y_axis_config={"numbers_to_include": [1, 2, 3, 4, 5]},
            tips=False,
        ).move_to([1.85, 0.15, 0.0])
        for ax in (axes.get_x_axis(), axes.get_y_axis()):
            ax.numbers.set_color(INK)
        x_title = mtex(r"t", font_size=36).move_to(axes.c2p(3.35, 0) + RIGHT * 0.3 + DOWN * 0.1)
        y_title = mtex(r"d(t)", font_size=36).move_to(axes.c2p(0, 5.3) + UP * 0.38)

        specs = [(0, FLAT, r"\kappa = 0"), (1, PRIMARY, r"\kappa = 1"), (-1, ACCENT, r"\kappa = -1")]
        graphs, tags = {}, {}
        for kappa, col, tag in specs:
            g = axes.plot(lambda t, k=kappa: spread(max(t, 1e-6), k),
                          x_range=[0, t_max, 0.01], stroke_color=col, stroke_width=6)
            graphs[kappa] = g
            tags[kappa] = mtex(tag, color=col, font_size=34)

        tags[-1].move_to(axes.c2p(2.35, 4.80))
        tags[0].move_to(axes.c2p(2.95, 2.45))
        tags[1].move_to(axes.c2p(2.42, 1.25))

        caption = mtex(
            r"\sec \ge \kappa \;\;\Longrightarrow\;\; "
            r"\text{geodesics in } M \text{ spread no faster than in } M_\kappa",
            font_size=38).move_to(DOWN * 3.45)

        self.play(FadeIn(schematic, run_time=1.2))
        self.play(FadeIn(VGroup(axes, x_title, y_title), run_time=0.9))
        self.play(Create(graphs[0], run_time=1.4), FadeIn(tags[0], run_time=1.4))
        self.play(Create(graphs[-1], run_time=1.4), FadeIn(tags[-1], run_time=1.4))
        self.play(Create(graphs[1], run_time=1.4), FadeIn(tags[1], run_time=1.4))
        self.wait(0.8)
        self.play(FadeIn(caption, shift=UP * 0.25, run_time=1.2))
        self.wait(3.8)


# ---------------------------------------------------------------------------
# Scene 3
# ---------------------------------------------------------------------------


class RoomInABall(Scene):
    """How many points, pairwise >= r apart, fit around the rim of a ball of radius r."""

    R_START, R_LOW, R_HIGH = 1.0, 0.5, 2.5

    def construct(self):
        disc_r = 1.45
        centres = [LEFT * 4.75 + UP * 0.2, UP * 0.2, RIGHT * 4.75 + UP * 0.2]
        specs = [(1, PRIMARY, r"\kappa > 0", "sphere"),
                 (0, FLAT, r"\kappa = 0", "flat"),
                 (-1, ACCENT, r"\kappa < 0", "hyperbolic")]

        tracker = ValueTracker(self.R_START)

        heads, subs, discs, rims, counts = (VGroup() for _ in range(5))

        for (kappa, col, head, name), c in zip(specs, centres):
            discs.add(VGroup(
                Circle(radius=disc_r, stroke_width=0, fill_color=col, fill_opacity=0.07),
                DashedVMobject(Circle(radius=disc_r, stroke_color=col, stroke_width=3),
                               num_dashes=72, dashed_ratio=0.5),
                Dot(ORIGIN, radius=0.055, color=col),
            ).move_to(c))
            heads.add(mtex(head, color=col, font_size=54).move_to(c + UP * 2.75))
            subs.add(label(name, font_size=28, color=GREY).move_to(c + UP * 2.18))

            rims.add(always_redraw(
                lambda k=kappa, cc=c, cl=col: self._rim(tracker.get_value(), k, cc, disc_r, cl)))
            counts.add(always_redraw(
                lambda k=kappa, cc=c, cl=col: label(
                    f"{rim_count(tracker.get_value(), k)}",
                    font_size=66, color=cl).move_to(cc + DOWN * 2.05)))

        r_read = always_redraw(lambda: label(
            f"ball of radius r = {tracker.get_value():.2f}", font_size=34, color=INK
        ).move_to(UP * 3.5))

        strap = label("points around the edge, pairwise ≥ r apart",
                      font_size=32, color=INK).move_to(DOWN * 3.4)
        caption = mtex(r"\kappa < 0:\ \text{room grows with } r,\ "
                       r"\text{so Theorem 1 needs } \operatorname{diam} \le D",
                       font_size=38).move_to(DOWN * 3.4)

        self.play(LaggedStart(*[FadeIn(h, shift=DOWN * 0.3) for h in heads],
                              lag_ratio=0.16, run_time=0.9),
                  LaggedStart(*[FadeIn(s) for s in subs], lag_ratio=0.16, run_time=0.9))
        self.play(LaggedStart(*[FadeIn(d) for d in discs], lag_ratio=0.16, run_time=0.9),
                  FadeIn(r_read, run_time=0.9))
        self.add(rims, counts)
        self.play(FadeIn(rims, run_time=1.0), FadeIn(counts, run_time=1.0))
        self.play(FadeIn(strap, run_time=0.7))
        self.wait(1.0)
        self.play(tracker.animate.set_value(self.R_LOW), run_time=1.2, rate_func=linear)
        self.play(tracker.animate.set_value(self.R_HIGH), run_time=5.5, rate_func=linear)
        self.wait(0.5)
        self.play(FadeOut(strap, run_time=0.5))
        self.play(FadeIn(caption, shift=UP * 0.25, run_time=1.1))
        self.wait(2.2)

    @staticmethod
    def _rim(r, kappa, centre, disc_r, col):
        """Rim points equally spaced at 360/N, with one gap marked."""
        n = rim_count(r, kappa)
        g = VGroup()
        ring = disc_r * 0.9
        angles = [math.pi / 2 + 2.0 * math.pi * k / n for k in range(n)]
        for a in angles:
            g.add(Dot(centre + ring * np.array([math.cos(a), math.sin(a), 0.0]),
                      radius=0.085, color=col))
        if n >= 2:
            step = 2.0 * math.pi / n
            a0, a1 = angles[0], angles[0] + step
            for a in (a0, a1):
                g.add(Line(centre,
                           centre + ring * np.array([math.cos(a), math.sin(a), 0.0]),
                           stroke_color=GREY, stroke_width=2, stroke_opacity=0.8))
            g.add(Arc(radius=0.42, start_angle=a0, angle=step, arc_center=centre,
                      stroke_color=INK, stroke_width=2.5))
            mid = a0 + step / 2
            g.add(label(f"{math.degrees(step):.0f}°", font_size=26, color=INK)
                  .move_to(centre + 0.74 * np.array([math.cos(mid), math.sin(mid), 0.0])))
        return g


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"{'r':>6} | {'kappa=0':>18} | {'kappa=1':>18} | {'kappa=-1':>18}")
    print("-" * 70)
    for i in range(21):
        r = 0.5 + i * 0.1
        row = f"{r:6.2f} |"
        for kappa in (0, 1, -1):
            a = min_angle(r, kappa)
            txt = "   --  " if a is None else f"{math.degrees(a):6.2f}°"
            row += f"  N={rim_count(r, kappa):>2}  need>{txt} |"
        print(row)
    print("\nself-check passed: flat 5 for all r; sphere 5 at r=1, 4 at r=1.5; "
          "hyperbolic 6 at r=1, 9 at r=2")
