"""
Sectional curvature via geodesic circles (Manim Community Edition).

Two scenes:
  1. CurvatureCircles  - grows a geodesic circle of radius r on a sphere (K = +1),
                         a plane (K = 0) and a saddle (K = -1 at its centre), and
                         shows circumference / (2*pi*r) for each.
  2. RoomInADisc       - plots C(r) for the three model geometries and links
                         "room in a disc" to the packing argument in Lemma 1.

Render (no LaTeX needed):
    pip install manim
    manim -pqm curvature.py CurvatureCircles
    manim -pqm curvature.py RoomInADisc
Use -pqh for high quality.
"""

import numpy as np
from manim import *

S = 1.25  # drawing scale (intrinsic units -> scene units)
R_MAX = 1.2  # largest geodesic radius shown
N_DIRS = 120  # directions shot out from the centre for the saddle


# ---------------------------------------------------------------------------
# Geometry helpers (all in intrinsic units, distances measured on the surface)
# ---------------------------------------------------------------------------


def saddle_f(x, y):
    """Saddle z = (x^2 - y^2)/2. Gaussian curvature at the origin is -1."""
    return 0.5 * (x**2 - y**2)


def saddle_geodesic_circle(r, n_dirs=N_DIRS, steps=60):
    """
    Points at geodesic distance r from the origin on the saddle, found by
    integrating the geodesic equation (RK4) in n_dirs directions.

    For a graph z = f(x,y) the geodesic equation is
        x'' = -f_x Q / W,  y'' = -f_y Q / W,
    with Q = f_xx x'^2 + 2 f_xy x' y' + f_yy y'^2 and W = 1 + f_x^2 + f_y^2.
    Here f_x = x, f_y = -y, f_xx = 1, f_yy = -1, f_xy = 0.
    """

    def acc(state):
        x, y, u, v = state
        Q = u**2 - v**2
        W = 1 + x**2 + y**2
        return np.array([u, v, -x * Q / W, y * Q / W])

    pts = []
    h = r / steps
    for th in np.linspace(0, 2 * np.pi, n_dirs, endpoint=False):
        # at the origin the surface is horizontal, so (cos, sin) is unit speed
        s = np.array([0.0, 0.0, np.cos(th), np.sin(th)])
        for _ in range(steps):
            k1 = acc(s)
            k2 = acc(s + h / 2 * k1)
            k3 = acc(s + h / 2 * k2)
            k4 = acc(s + h * k3)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x, y = s[0], s[1]
        pts.append(np.array([x, y, saddle_f(x, y)]))
    return np.array(pts)


def sphere_circle(r, n=N_DIRS):
    """Geodesic circle of radius r about the north pole of the unit sphere."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack(
        [np.sin(r) * np.cos(th), np.sin(r) * np.sin(th), np.full_like(th, np.cos(r))],
        axis=1,
    )


def plane_circle(r, n=N_DIRS):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)], axis=1)


def loop_length(pts):
    return np.sum(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1))


# ---------------------------------------------------------------------------
# Scene 1: geodesic circles on three surfaces
# ---------------------------------------------------------------------------


class CurvatureCircles(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=62 * DEGREES, theta=-90 * DEGREES, zoom=0.85)

        centres = {"sphere": LEFT * 4.6, "plane": ORIGIN, "saddle": RIGHT * 4.6}

        # --- surfaces ------------------------------------------------------
        sphere = Surface(
            lambda u, v: (
                S * np.array([np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v)])
                + centres["sphere"]
            ),
            u_range=[0, 2 * PI],
            v_range=[0, PI],
            resolution=(28, 14),
        )
        plane = Surface(
            lambda u, v: S * np.array([u, v, 0]) + centres["plane"],
            u_range=[-1.45, 1.45],
            v_range=[-1.45, 1.45],
            resolution=(10, 10),
        )
        saddle = Surface(
            lambda u, v: S * np.array([u, v, saddle_f(u, v)]) + centres["saddle"],
            u_range=[-1.45, 1.45],
            v_range=[-1.45, 1.45],
            resolution=(20, 20),
        )
        # lift the plane and saddle so their centres sit level with the sphere's pole
        plane.shift(S * OUT)
        saddle.shift(S * OUT)

        for surf, col in [(sphere, BLUE_E), (plane, GREY_D), (saddle, TEAL_E)]:
            surf.set_fill_by_checkerboard(
                col, interpolate_color(col, WHITE, 0.15), opacity=0.55
            )
            surf.set_stroke(width=0.3, opacity=0.4)

        titles = (
            VGroup(
                Text("Sphere  (K > 0)", font_size=28, color=BLUE_B),
                Text("Plane  (K = 0)", font_size=28, color=GREY_B),
                Text("Saddle  (K < 0)", font_size=28, color=TEAL_B),
            )
            .arrange(RIGHT, buff=1.9)
            .to_edge(UP, buff=0.35)
        )
        self.add_fixed_in_frame_mobjects(titles)

        self.play(
            FadeIn(titles), Create(sphere), Create(plane), Create(saddle), run_time=2
        )

        # --- growing geodesic circles --------------------------------------
        r = ValueTracker(0.05)
        centre_dots = VGroup(
            Dot(centres["sphere"] + S * OUT, radius=0.05, color=YELLOW),
            Dot(centres["plane"] + S * OUT, radius=0.05, color=YELLOW),
            Dot(centres["saddle"] + S * OUT, radius=0.05, color=YELLOW),
        )
        self.play(FadeIn(centre_dots))

        def to_scene(pts, key):
            return S * pts + centres[key] + (S * OUT if key != "sphere" else 0)

        def make_loop(pts, color):
            loop = VMobject(stroke_color=color, stroke_width=5, fill_opacity=0)
            loop.set_points_smoothly([*pts, pts[0]])
            return loop

        cols = [BLUE_B, GREY_B, TEAL_B]

        def build():
            """Rebuild the three circles and the text readouts for the current r."""
            rv = r.get_value()
            sp, pl, sa = sphere_circle(rv), plane_circle(rv), saddle_geodesic_circle(rv)
            loops = VGroup(
                make_loop(to_scene(sp, "sphere"), YELLOW),
                make_loop(to_scene(pl, "plane"), YELLOW),
                make_loop(to_scene(sa, "saddle"), YELLOW),
            )
            flat = 2 * np.pi * rv
            ratios = [loop_length(p) / flat for p in (sp, pl, sa)]
            readouts = VGroup()
            for q, c, ttl in zip(ratios, cols, titles):
                t = Text(f"C / 2πr = {q:.3f}", font_size=26, color=c)
                t.next_to(ttl, DOWN, buff=0.2)
                readouts.add(t)
            rtext = Text(f"geodesic radius r = {rv:.2f}", font_size=28)
            rtext.to_edge(DOWN, buff=0.4)
            return loops, VGroup(readouts, rtext)

        camera = self.renderer.camera
        loop_holder, text_holder = VGroup(), VGroup()

        def update_loops(holder):
            holder.remove(*holder.submobjects)
            holder.add(build()[0])

        def update_text(holder):
            holder.remove(*holder.submobjects)
            holder.add(build()[1])
            camera.add_fixed_in_frame_mobjects(holder)  # keep new text flat on screen

        # Holders start empty: Manim's Cairo renderer snapshots the family of
        # moving mobjects when play() begins, so any children present then
        # would be drawn as frozen "ghosts" for the rest of the animation.
        loop_holder.add_updater(update_loops)
        text_holder.add_updater(update_text)
        self.add(loop_holder, text_holder)

        self.play(r.animate.set_value(R_MAX), run_time=7, rate_func=linear)
        loop_holder.clear_updaters()
        text_holder.clear_updaters()
        self.wait(0.5)

        msg = Text(
            "Sphere: circles shorter than 2πr  ·  Plane: exactly 2πr  ·  "
            "Saddle: longer than 2πr",
            font_size=24,
        ).to_edge(DOWN, buff=0.4)
        text_holder[0].remove(text_holder[0][1])  # drop the "r = ..." line
        self.add_fixed_in_frame_mobjects(msg)
        self.play(FadeIn(msg))
        self.begin_ambient_camera_rotation(rate=0.06)
        self.wait(4)
        self.stop_ambient_camera_rotation()
        self.wait(0.5)


# ---------------------------------------------------------------------------
# Scene 2: "room in a disc" and the link to packing
# ---------------------------------------------------------------------------


class RoomInADisc(Scene):
    def construct(self):
        title = Text("How much room is there at distance r?", font_size=36)
        title.to_edge(UP, buff=0.4)

        axes = (
            Axes(
                x_range=[0, 2.2, 0.5],
                y_range=[0, 30, 10],
                x_length=5.6,
                y_length=4.4,
                axis_config={"include_numbers": False, "include_tip": True},
            )
            .to_edge(LEFT, buff=0.8)
            .shift(DOWN * 0.3)
        )
        xl = Text("r", font_size=28).next_to(axes.x_axis, RIGHT, buff=0.15)
        yl = Text("circumference C(r)", font_size=24).next_to(
            axes.y_axis, UP, aligned_edge=LEFT
        )

        neg = axes.plot(lambda t: 2 * np.pi * np.sinh(t), x_range=[0, 2.2], color=TEAL)
        flat = axes.plot(lambda t: 2 * np.pi * t, x_range=[0, 2.2], color=GREY_B)
        pos = axes.plot(lambda t: 2 * np.pi * np.sin(t), x_range=[0, 2.2], color=BLUE)

        legend = VGroup(
            Text("K = −1:  2π sinh r", font_size=24, color=TEAL),
            Text("K = 0:   2π r", font_size=24, color=GREY_B),
            Text("K = +1:  2π sin r", font_size=24, color=BLUE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)

        notes = VGroup(
            Text("More circumference", font_size=24),
            Text("⇒ more room in the disc", font_size=24),
            Text("⇒ more points fit pairwise ≥ r apart", font_size=24),
            Text("⇒ larger packing number P(M)", font_size=24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        column = VGroup(legend, notes).arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        column.next_to(axes, RIGHT, buff=0.7).align_to(axes, UP)
        if column.get_right()[0] > config.frame_x_radius - 0.3:
            column.scale_to_fit_width(
                config.frame_x_radius - 0.3 - column.get_left()[0], about_edge=LEFT
            )

        self.play(Write(title))
        self.play(Create(axes), FadeIn(xl), FadeIn(yl))
        for curve, lab in zip([flat, pos, neg], [legend[1], legend[2], legend[0]]):
            self.play(Create(curve), FadeIn(lab), run_time=1.5)
        self.wait(0.5)

        for line in notes:
            self.play(FadeIn(line, shift=0.2 * UP), run_time=0.8)
        self.wait(0.5)

        punch = Text(
            "sec ≥ κ caps the room  ⇒  caps P(M)  ⇒  |NND| ≤ P(M) + 1",
            font_size=28,
            color=YELLOW,
        ).to_edge(DOWN, buff=0.35)
        box = SurroundingRectangle(punch, color=YELLOW, buff=0.15)
        self.play(Write(punch), Create(box))
        self.wait(3)
