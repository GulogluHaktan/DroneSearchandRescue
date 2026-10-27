"""Write one mission (env index e of a World) into a USD stage: terrain mesh, point-instanced trunks /
foliage clumps / bushes, and box-shaped persons with the semantic label "person".

pxr only (runs inside Isaac Sim); the same primitives as the torch ray caster, so both layers see
the same geometry. Photoreal tree / human assets are a funded-phase item (G7 of the review).
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdSemantics, Vt

ROOT = "/World/Forest"
TRUNK_RGB = (0.23, 0.16, 0.10)
CONIFER_RGB = (0.07, 0.20, 0.08)
BROAD_RGB = (0.13, 0.30, 0.09)
BUSH_RGB = (0.20, 0.33, 0.12)
GROUND_RGB = (0.24, 0.22, 0.14)
PERSON_RGB = (0.95, 0.35, 0.05)  # orange jacket


def _color(prim, rgb):
    UsdGeom.Gprim(prim).CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*rgb)]))


def _instancer(stage, path, protos, positions, scales, proto_idx):
    inst = UsdGeom.PointInstancer.Define(stage, path)
    rel = inst.CreatePrototypesRel()
    for p in protos:
        rel.AddTarget(p)
    inst.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(positions.astype(np.float32)))
    inst.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(scales.astype(np.float32)))
    inst.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(proto_idx.astype(np.int32)))
    return inst


def export(stage, world, e: int = 0, lights: bool = True) -> dict:
    """Returns paths of the occluder prims (to hide them) and of the persons."""
    UsdGeom.Xform.Define(stage, ROOT)
    # terrain
    h = world.terrain.h[e].cpu().numpy()
    n, cell = h.shape[0], world.terrain.cell
    xs = np.arange(n) * cell
    X, Y = np.meshgrid(xs, xs)
    pts = np.stack((X, Y, h), -1).reshape(-1, 3).astype(np.float32)
    idx = np.arange(n * n).reshape(n, n)
    quads = np.stack((idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]), -1).reshape(-1)
    mesh = UsdGeom.Mesh.Define(stage, f"{ROOT}/Terrain")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * ((n - 1) * (n - 1))))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(quads.astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr("none")
    _color(mesh.GetPrim(), GROUND_RGB)

    f = world.forests_np[e]
    UsdGeom.Xform.Define(stage, f"{ROOT}/Occluders")

    def protos_under(inst_path, specs):
        """Prototypes as children of the instancer: USD images them only through the instancer."""
        out = []
        for name, kind, rgb in specs:
            g = (UsdGeom.Cylinder if kind == "cyl" else UsdGeom.Sphere).Define(stage, f"{inst_path}/{name}")
            g.CreateRadiusAttr(1.0)
            if kind == "cyl":
                g.CreateHeightAttr(1.0)
                g.CreateAxisAttr("Z")
            _color(g.GetPrim(), rgb)
            out.append(g.GetPath())
        return out

    ht = 0.9 * f.tree_h
    pos = np.concatenate((f.tree_xy, (f.tree_z0 + ht / 2)[:, None]), -1)
    sc = np.stack((f.trunk_r, f.trunk_r, ht), -1)
    p_ = f"{ROOT}/Occluders/Trunks"
    UsdGeom.PointInstancer.Define(stage, p_)
    _instancer(stage, p_, protos_under(p_, [("Trunk", "cyl", TRUNK_RGB)]), pos, sc, np.zeros(len(pos)))
    cl = f.clumps.reshape(-1, 4)
    kind = np.repeat(np.where(f.conifer, 0, 1), f.clumps.shape[1])
    p_ = f"{ROOT}/Occluders/Foliage"
    UsdGeom.PointInstancer.Define(stage, p_)
    pr = protos_under(p_, [("Conifer", "sph", CONIFER_RGB), ("Broadleaf", "sph", BROAD_RGB)])
    _instancer(stage, p_, pr, cl[:, :3], np.repeat(cl[:, 3:4], 3, 1), kind)
    if len(f.bushes):
        p_ = f"{ROOT}/Occluders/Bushes"
        UsdGeom.PointInstancer.Define(stage, p_)
        _instancer(stage, p_, protos_under(p_, [("Bush", "sph", BUSH_RGB)]), f.bushes[:, :3],
                   np.repeat(f.bushes[:, 3:4], 3, 1), np.zeros(len(f.bushes)))

    persons = []
    p = world.persons_np[e]
    for i in range(len(p.pos)):
        path = f"{ROOT}/Person{i}"
        xf = UsdGeom.Xform.Define(stage, path)
        l, w, hh = (float(v) for v in p.dims[i])
        c = p.pos[i]
        xf.AddTranslateOp().Set(Gf.Vec3d(float(c[0]), float(c[1]), float(c[2]) + hh / 2))
        xf.AddRotateZOp().Set(math.degrees(float(p.yaw[i])))
        body = UsdGeom.Cube.Define(stage, f"{path}/Body")
        body.CreateSizeAttr(1.0)
        UsdGeom.Xformable(body).AddScaleOp().Set(Gf.Vec3f(l, w, hh))
        _color(body.GetPrim(), PERSON_RGB)
        for prim in (xf.GetPrim(), body.GetPrim()):
            UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr().Set(["person"])
        persons.append(path)

    if lights:
        dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
        dome.CreateIntensityAttr(600.0)
        dome.CreateColorAttr(Gf.Vec3f(0.75, 0.85, 1.0))
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(2500.0)
        sun.CreateAngleAttr(0.53)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 20.0, 40.0))
    return {"occluders": f"{ROOT}/Occluders", "persons": persons, "root": ROOT}


def set_visible(stage, path: str, visible: bool) -> None:
    im = UsdGeom.Imageable(stage.GetPrimAtPath(Sdf.Path(path)))
    im.MakeVisible() if visible else im.MakeInvisible()
