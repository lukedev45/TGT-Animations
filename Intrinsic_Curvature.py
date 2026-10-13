"""
Intrinsic curvature: what an ant living on the surface can measure.

Two scenes (Manim Community Edition, no LaTeX needed):

  1. BendingWithoutStretching - a flat sheet rolls into a cylinder. The normal
     vectors turn (extrinsic curvature changes), but a circle and a triangle
     drawn on the sheet keep C / 2πr = 1 and an angle sum of 180°: the
     intrinsic curvature stays K = 0.

  2. SphereTriangle - a geodesic triangle grows on the unit sphere. Its angle
     sum climbs above 180°, something an ant can measure with a protractor
     without ever leaving the surface. At the octant triangle (three right
     angles) the excess equals K × area (Gauss-Bonnet).

Render:
    pip install manim
    manim -pqm intrinsic_curvature.py BendingWithoutStretching
    manim -pqm intrinsic_curvature.py SphereTriangle
Use -pqh for 1080p.
"""

import numpy as np
from manim import *


def loop_length(pts):
    return np.sum(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1))


def angle_between(u, v):
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return np.degrees(np.arccos(np.clip(np.dot(u, v), -1, 1)))


# ---------------------------------------------------------------------------
# Scene 1: rolling a sheet into a cylinder
# ---------------------------------------------------------------------------

L, H = 2.6, 1.6  # sheet is [-L, L] x [-H, H]
C_FINAL = 1.15  # final bending curvature (cylinder radius 1/C_FINAL)

CIRCLE_CENTRE = np.array([-1.05, 0.2])
CIRCLE_R = 0.75
TRIANGLE = [np.array([0.2, -1.0]), np.array([1.75, -1.0]), np.array([0.95, 0.95])]
OFFSET = np.array([-2.7, 0.0, 0.0])  # pushes the sheet to the left of the frame
NORMAL_XS = np.linspace(-2.4, 2.4, 7)
NORMAL_Y = 1.45


def bend(x, y, c):
    """
    Roll the sheet about an axis parallel to y. Arc length along x is kept,
    so this map is an isometry for every c: it bends but never stretches.
    """
    if abs(c) < 1e-6:
        return np.array([x, y, 0.0]) + OFFSET
    return np.array([np.sin(c * x) / c, y, -(1 - np.cos(c * x)) / c]) + OFFSET


def bent_normal(x, c):
    """Outward unit normal of the bent sheet at horizontal position x."""
    return np.array([np.sin(c * x), 0.0, np.cos(c * x)])


class BendingWithoutStretching(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=64 * DEGREES, theta=-78 * DEGREES, zoom=1.12)
        cam = self.renderer.camera

        title = Text("Bending without stretching", font_size=34).to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)
        self.play(FadeIn(title))

        c = ValueTracker(0.0)

        def build_scene_objects():
            cv = c.get_value()
            sheet = Surface(
                lambda u, v: bend(u, v, cv),
                u_range=[-L, L],
                v_range=[-H, H],
                resolution=(26, 8),
            )
            sheet.set_fill_by_checkerboard(GREY_D, GREY_C, opacity=0.6)
            sheet.set_stroke(width=0.3, opacity=0.4)

            th = np.linspace(0, 2 * np.pi, 160, endpoint=False)
            circ_pts = np.array(
                [
                    bend(
                        CIRCLE_CENTRE[0] + CIRCLE_R * np.cos(t),
                        CIRCLE_CENTRE[1] + CIRCLE_R * np.sin(t),
                        cv,
                    )
                    for t in th
                ]
            )
            circle = VMobject(stroke_color=YELLOW, stroke_width=5, fill_opacity=0)
            circle.set_points_smoothly([*circ_pts, circ_pts[0]])

            # straight sides in sheet coordinates = geodesics on the bent sheet
            tri = VGroup()
            for i in range(3):
                a, b = TRIANGLE[i], TRIANGLE[(i + 1) % 3]
                pts = [bend(*(a + s * (b - a)), cv) for s in np.linspace(0, 1, 40)]
                side = VMobject(stroke_color=ORANGE, stroke_width=5, fill_opacity=0)
                side.set_points_smoothly(pts)
                tri.add(side)

            normals = VGroup()
            for x in NORMAL_XS:
                base = bend(x, NORMAL_Y, cv)
                tip = base + 0.7 * bent_normal(x, cv)
                normals.add(
                    Line(base, tip, color=RED_B, stroke_width=4),
                    Dot(tip, radius=0.045, color=RED_B),
                )

            # intrinsic measurements, done on the bent surface in 3D
            ratio = loop_length(circ_pts) / (2 * np.pi * CIRCLE_R)
            eps = 1e-4
            angle_sum = 0.0
            for i in range(3):
                v = TRIANGLE[i]
                p, q = TRIANGLE[(i + 1) % 3], TRIANGLE[(i + 2) % 3]
                tp = bend(*(v + eps * (p - v)), cv) - bend(*v, cv)
                tq = bend(*(v + eps * (q - v)), cv) - bend(*v, cv)
                angle_sum += angle_between(tp, tq)

            return VGroup(sheet, circle, tri, normals), (cv, ratio, angle_sum)

        def build_text(cv, ratio, angle_sum):
            ext = VGroup(
                Text("Extrinsic  (seen from outside)", font_size=24, color=RED_B),
                Text(f"normals turn:  κ₁ = {cv:.2f},  κ₂ = 0", font_size=22),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            intr = VGroup(
                Text("Intrinsic  (measured on the sheet)", font_size=24, color=YELLOW),
                Text(f"circle:  C / 2πr = {ratio:.3f}", font_size=22),
                Text(f"triangle:  angle sum = {angle_sum:.1f}°", font_size=22),
                Text("K = κ₁ κ₂ = 0", font_size=22),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            panel = VGroup(ext, intr).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
            panel.to_edge(RIGHT, buff=0.4).shift(UP * 0.6)
            return panel

        # Holders start empty: Manim's Cairo renderer snapshots the family of
        # moving mobjects when play() begins, so children present then would be
        # drawn as frozen ghosts for the rest of the animation.
        geom_holder, text_holder = VGroup(), VGroup()
        latest = {}

        def update_geom(holder):
            objs, nums = build_scene_objects()
            latest["nums"] = nums
            holder.remove(*holder.submobjects)
            holder.add(objs)

        def update_text(holder):
            holder.remove(*holder.submobjects)
            holder.add(build_text(*latest["nums"]))
            cam.add_fixed_in_frame_mobjects(holder)

        geom_holder.add_updater(update_geom)
        text_holder.add_updater(update_text)
        self.add(geom_holder, text_holder)

        def fresh():
            # Empty the holders before every play() while they update.
            geom_holder.remove(*geom_holder.submobjects)
            text_holder.remove(*text_holder.submobjects)

        def fill():
            # wait() renders one static frame (our updaters aren't time-based),
            # so the holders must already contain the current picture.
            update_geom(geom_holder)
            update_text(text_holder)

        # flat sheet first, then roll it up, then unroll a little and roll again
        fill()
        self.wait(1.5)
        fresh()
        self.play(c.animate.set_value(C_FINAL), run_time=5, rate_func=smooth)
        fill()
        self.wait(1)
        fresh()
        self.play(c.animate.set_value(0.45), run_time=2, rate_func=smooth)
        fresh()
        self.play(c.animate.set_value(C_FINAL), run_time=2, rate_func=smooth)
        geom_holder.clear_updaters()
        text_holder.clear_updaters()

        verdict = Text(
            "The shape in space changed. The geometry an ant can measure did not.",
            font_size=26,
            color=YELLOW,
        ).to_edge(DOWN, buff=0.35)
        self.add_fixed_in_frame_mobjects(verdict)
        self.play(FadeIn(verdict))
        self.begin_ambient_camera_rotation(rate=0.08)
        self.wait(4)
        self.stop_ambient_camera_rotation()


# ---------------------------------------------------------------------------
# Scene 2: geodesic triangles on the sphere
# ---------------------------------------------------------------------------

S = 2.2  # drawing radius of the unit sphere
NORTH = np.array([0.0, 0.0, 1.0])
OCTANT_RHO = np.arccos(1 / np.sqrt(3))  # circumradius of the 90-90-90 triangle


def slerp(a, b, t):
    om = np.arccos(np.clip(np.dot(a, b), -1, 1))
    if om < 1e-9:
        return a
    return (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / np.sin(om)


def tangent_towards(v, p):
    t = p - np.dot(v, p) * v
    return t / np.linalg.norm(t)


def triangle_vertices(rho):
    """Equilateral geodesic triangle centred on the north pole, circumradius rho."""
    return [
        np.array([np.sin(rho) * np.cos(a), np.sin(rho) * np.sin(a), np.cos(rho)])
        for a in (np.pi / 6, np.pi / 6 + 2 * np.pi / 3, np.pi / 6 + 4 * np.pi / 3)
    ]


class SphereTriangle(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=38 * DEGREES, theta=-60 * DEGREES, zoom=1.0)
        cam = self.renderer.camera

        title = Text("Measuring curvature from inside", font_size=32)
        title.to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)

        sphere = Surface(
            lambda u, v: (
                S * np.array([np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v)])
            ),
            u_range=[0, 2 * PI],
            v_range=[0, PI],
            resolution=(32, 16),
        )
        sphere.set_fill_by_checkerboard(BLUE_E, BLUE_D, opacity=0.5)
        sphere.set_stroke(width=0.3, opacity=0.35)
        self.play(FadeIn(title), Create(sphere), run_time=2)

        rho = ValueTracker(0.12)

        def build_triangle():
            verts = triangle_vertices(rho.get_value())
            tri = VGroup()
            for i in range(3):
                a, b = verts[i], verts[(i + 1) % 3]
                pts = [S * 1.005 * slerp(a, b, t) for t in np.linspace(0, 1, 50)]
                side = VMobject(stroke_color=ORANGE, stroke_width=5, fill_opacity=0)
                side.set_points_smoothly(pts)
                tri.add(side)

            angles, arcs = [], VGroup()
            for i in range(3):
                v = verts[i]
                t1 = tangent_towards(v, verts[(i + 1) % 3])
                t2 = tangent_towards(v, verts[(i + 2) % 3])
                ang = np.arccos(np.clip(np.dot(t1, t2), -1, 1))
                angles.append(np.degrees(ang))
                # small arc on the sphere marking the angle at v
                u = t2 - np.dot(t2, t1) * t1
                u /= np.linalg.norm(u)
                eps = 0.16
                pts = [
                    S
                    * 1.008
                    * (np.cos(eps) * v + np.sin(eps) * (np.cos(s) * t1 + np.sin(s) * u))
                    for s in np.linspace(0, ang, 24)
                ]
                arc = VMobject(stroke_color=YELLOW, stroke_width=4, fill_opacity=0)
                arc.set_points_smoothly(pts)
                arcs.add(arc)
            return VGroup(tri, arcs), angles

        def build_text(angles):
            total = sum(angles)
            panel = VGroup(
                Text(f"each angle = {angles[0]:.1f}°", font_size=26),
                Text(f"angle sum = {total:.1f}°", font_size=30, color=YELLOW),
                Text("on a flat plane: always 180°", font_size=22, color=GREY_B),
                Text(f"excess = {total - 180:.1f}°", font_size=26, color=ORANGE),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            panel.next_to(title, DOWN, buff=0.5, aligned_edge=LEFT)
            return panel

        geom_holder, text_holder = VGroup(), VGroup()
        latest = {}

        def update_geom(holder):
            objs, angles = build_triangle()
            latest["angles"] = angles
            holder.remove(*holder.submobjects)
            holder.add(objs)

        def update_text(holder):
            holder.remove(*holder.submobjects)
            holder.add(build_text(latest["angles"]))
            cam.add_fixed_in_frame_mobjects(holder)

        geom_holder.add_updater(update_geom)
        text_holder.add_updater(update_text)
        self.add(geom_holder, text_holder)

        def fresh():
            # Empty the holders before every play() while they update.
            geom_holder.remove(*geom_holder.submobjects)
            text_holder.remove(*text_holder.submobjects)

        def fill():
            # wait() renders one static frame (our updaters aren't time-based),
            # so the holders must already contain the current picture.
            update_geom(geom_holder)
            update_text(text_holder)

        # small triangle: almost flat, angle sum ≈ 180°
        fill()
        self.wait(1.5)
        # grow to the octant triangle: three right angles
        fresh()
        self.play(rho.animate.set_value(OCTANT_RHO), run_time=5, rate_func=smooth)
        fill()
        self.wait(0.5)

        gb = (
            VGroup(
                Text("Three right angles: excess 90° = π/2", font_size=22),
                Text("Area = π/2  (one eighth of the sphere)", font_size=22),
                Text(
                    "Gauss–Bonnet: excess = K × area, so K = 1",
                    font_size=24,
                    color=YELLOW,
                ),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            .to_corner(DR, buff=0.4)
        )
        self.add_fixed_in_frame_mobjects(gb)
        fresh()
        self.play(FadeIn(gb, shift=0.2 * UP), run_time=1.5)
        fill()
        self.wait(3)
        fresh()
        self.play(FadeOut(gb))
        cam.remove_fixed_in_frame_mobjects(gb)

        # keep growing: the more area, the more curvature adds up
        fresh()
        self.play(rho.animate.set_value(1.45), run_time=4, rate_func=smooth)
        geom_holder.clear_updaters()
        text_holder.clear_updaters()

        verdict = Text(
            "Measured entirely on the surface: no normals, no outside space.",
            font_size=26,
            color=YELLOW,
        ).to_edge(DOWN, buff=0.35)
        self.add_fixed_in_frame_mobjects(verdict)
        self.play(FadeIn(verdict))
        self.wait(3)
