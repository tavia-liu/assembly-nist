"""One MJCF with several parts physically on the NIST board at once.

Each task's single-part scene (assembly_policy.sim.scene.build_xml) contributes its exact fixture body and its free
part body; task-specific names (bodies, joints, sites, geoms, mesh assets, SDF plugin instances) get a `_<task>`
suffix so they coexist. The base scene is a fine-physics task when there is one (0.5 ms step for gears / nuts),
so the common timestep is the finest required. Remaining fixtures stay static convex hulls as in the single-part
scene. The three gears share one base: it is added once, and the static meshes it carries for the other gears
are dropped for every gear that is a movable part of the scene.
"""
from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET

from assembly_policy.sim.gears import GEARS, gear_base_geoms, gear_convex_pieces
from assembly_policy.sim.scene import PhysicsConfig, build_xml, task_physics
from assembly_policy.sim.tasks import TASKS


def _rename_task(root, t):
    """Suffix the task-specific names of one single-part scene with _<t>."""
    asset, world = root.find("asset"), root.find("worldbody")
    mesh_map = {}
    for bname in ("fixture", "part"):
        body = world.find(f"body[@name='{bname}']")
        body.set("name", f"{bname}_{t}")
        for site in body.findall("site"):
            site.set("name", f"{site.get('name')}_{t}")
        for j in body.findall("freejoint"):
            j.set("name", f"{j.get('name')}_{t}")
        for g in body.findall("geom"):
            if g.get("name"):
                g.set("name", f"{g.get('name')}_{t}")
            m = g.get("mesh")
            if m and m.startswith(("fix", "part")) and m.endswith("_mesh"):
                mesh_map[m] = f"{m[:-5]}_{t}_mesh"; g.set("mesh", mesh_map[m])
            elif m and m.startswith("sdf_"):
                mesh_map[m] = f"{m}_{t}"; g.set("mesh", mesh_map[m])
            for p in g.findall("plugin"):
                p.set("instance", f"{p.get('instance')}_{t}")
    for m in asset.findall("mesh"):
        if m.get("name") in mesh_map:
            m.set("name", mesh_map[m.get("name")])
            for p in m.findall("plugin"):
                p.set("instance", f"{p.get('instance')}_{t}")
    ext = root.find("extension")
    if ext is not None:
        for inst in ext.iter("instance"):
            inst.set("name", f"{inst.get('name')}_{t}")


def _prune_gear_base(root, t, gear_parts):
    """Drop the static meshes of gears that are movable parts of this scene from the gear base of task t."""
    body = root.find("worldbody").find(f"body[@name='fixture_{t}']")
    asset = root.find("asset")
    k = len(gear_base_geoms())
    for other in GEARS:
        if other == TASKS[t].part_asset:
            continue
        n = len(gear_convex_pieces(other))
        if other in gear_parts:
            for j in range(k, k + n):
                body.remove(body.find(f"geom[@name='fix{j}_{t}']"))
                asset.remove(asset.find(f"mesh[@name='fix{j}_{t}_mesh']"))
        k += n


def build_multi_xml(tasks, cfg: PhysicsConfig | None = None, visual: bool = False, physics_of: str | None = None) -> str:
    """physics_of=<task>: the whole scene uses that task's physics (its timestep and contact parameters; the other
    parts are obstacles only) -- e.g. a coarse 2.5 ms scene for training on P4 with gears and nuts lying around."""
    tasks = list(tasks)
    assert len(set(tasks)) == len(tasks), tasks
    gear_parts = {TASKS[t].part_asset for t in tasks if TASKS[t].family == "gear"}
    cfg = cfg or PhysicsConfig(fine_families="gear+nut")
    if physics_of is not None:
        cfg = dataclasses.replace(task_physics(physics_of, cfg), fine_families="")
        base = physics_of
    else:
        fine = {f for f in cfg.fine_families.replace(",", "+").split("+") if f}
        base = next((t for t in tasks if TASKS[t].family in fine), tasks[0])
    roots = {t: ET.fromstring(build_xml(t, cfg, visual)) for t in tasks}
    root = roots[base]
    root.set("model", "assemblenist_" + "_".join(tasks))
    _rename_task(root, base)
    asset, world, opt = root.find("asset"), root.find("worldbody"), root.find("option")
    ext = root.find("extension")
    have = {m.get("name") for m in asset.findall("mesh")}
    fixtures = {TASKS[base].fixture_scene: world.find(f"body[@name='fixture_{base}']")}
    if TASKS[base].family == "gear":
        _prune_gear_base(root, base, gear_parts)
    for t in tasks:
        if t == base:
            continue
        r = roots[t]
        _rename_task(r, t)
        if TASKS[t].family == "gear":
            _prune_gear_base(r, t, gear_parts)
        rext = r.find("extension")
        if rext is not None:
            if ext is None:
                ext = ET.SubElement(root, "extension")
            for pe in rext.findall("plugin"):
                tgt = ext.find(f"plugin[@plugin='{pe.get('plugin')}']")
                if tgt is None:
                    tgt = ET.SubElement(ext, "plugin", plugin=pe.get("plugin"))
                for inst in pe.findall("instance"):
                    tgt.append(inst)
            ropt = r.find("option")
            for k in ("sdf_initpoints", "sdf_iterations"):
                if ropt.get(k):
                    opt.set(k, str(max(int(opt.get(k, "0")), int(ropt.get(k)))))
        for m in r.find("asset").findall("mesh"):
            if m.get("name") not in have:
                asset.append(m); have.add(m.get("name"))
        world.append(r.find("worldbody").find(f"body[@name='part_{t}']"))
        fscene = TASKS[t].fixture_scene
        if fscene in fixtures:                       # shared fixture (the gear base): one body, one site per task
            ET.SubElement(fixtures[fscene], "site", name=f"fixture_frame_{t}", size="0.002", group="4")
        else:
            fixtures[fscene] = r.find("worldbody").find(f"body[@name='fixture_{t}']")
            world.append(fixtures[fscene])
        for g in list(world.findall("geom")):       # this task's fixture was a static hull in the base scene
            if g.get("name") in (f"obst_{TASKS[t].fixture_scene}", f"vis_obst_{TASKS[t].fixture_scene}"):
                world.remove(g)
    return ET.tostring(root, encoding="unicode")
