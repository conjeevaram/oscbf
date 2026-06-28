"""MJCF (MuJoCo XML) parser for oscbf.

Output format matches parse_urdf — same keys, same conventions — so
Manipulator.from_mjcf is a drop-in for Manipulator.from_urdf.

Fixed bodies (no hinge/slide joint) merge their inertia into the nearest
moveable ancestor via the parallel-axis theorem.  inertia_rot is always
identity in the output, consistent with the URDF parser convention.
"""

import xml.etree.ElementTree as ET
import numpy as np


def _parse_floats(s):
    return [float(x) for x in s.split()]


def _quat_to_matrix(wxyz):
    # MJCF quaternion convention: (w, x, y, z)
    q = np.asarray(wxyz, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ])


def _fullinertia_to_matrix(vals):
    ixx, iyy, izz, ixy, ixz, iyz = vals
    return np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ])


def _steiner(I, m, d):
    """Parallel-axis (Steiner) shift of inertia tensor I by displacement d."""
    return I + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


def _merge_inertia(m1, com1, I1, m2, com2, I2):
    """Merge two inertia tensors; all inputs in the same reference frame."""
    total_mass = m1 + m2
    combined_com = (m1 * com1 + m2 * com2) / total_mass if total_mass > 0 else com1.copy()
    combined_I = (
        _steiner(I1, m1, combined_com - com1)
        + _steiner(I2, m2, combined_com - com2)
    )
    return total_mass, combined_com, combined_I


def _build_defaults(root):
    # {class_name: {"_parent": str|None, element_tag: {attr: val, ...}, ...}}
    defaults = {}

    def _parse_default(elem, parent_class):
        class_name = elem.get("class")
        if class_name is None:
            for child in elem:
                if child.tag == "default":
                    _parse_default(child, parent_class)
            return

        if class_name not in defaults:
            defaults[class_name] = {"_parent": parent_class}
        else:
            defaults[class_name].setdefault("_parent", parent_class)

        for child in elem:
            if child.tag == "default":
                _parse_default(child, parent_class=class_name)
            else:
                tag = child.tag
                if tag not in defaults[class_name]:
                    defaults[class_name][tag] = {}
                defaults[class_name][tag].update(child.attrib)

    default_section = root.find("default")
    if default_section is not None:
        _parse_default(default_section, parent_class=None)

    return defaults


def _resolve_attrs(elem_tag, elem_attrs, class_name, defaults):
    # Walk class chain root→nearest; element's own attrs always win.
    chain = []
    c = class_name
    visited = set()
    while c is not None and c not in visited:
        visited.add(c)
        chain.append(c)
        c = defaults.get(c, {}).get("_parent")

    result = {}
    for c in reversed(chain):
        if c in defaults and elem_tag in defaults[c]:
            result.update(defaults[c][elem_tag])
    result.update(elem_attrs)
    return result


def _parse_actuators(root, defaults):
    # Returns {joint_name: (force_lo, force_hi)} from the <actuator> section.
    force_limits = {}
    actuator_section = root.find("actuator")
    if actuator_section is None:
        return force_limits

    for act in actuator_section:
        joint_name = act.get("joint")
        if joint_name is None:
            continue
        eff = _resolve_attrs(act.tag, act.attrib, act.get("class"), defaults)
        if "forcerange" in eff:
            lo, hi = _parse_floats(eff["forcerange"])
            force_limits[joint_name] = (lo, hi)

    return force_limits


def _parse_inertial(inertial_elem, effective_class, defaults):
    # Returns (mass, com_pos, inertia_3x3) in the body's own frame.
    # diaginertia+quat rotation is absorbed into the tensor; caller sets inertia_rot = I.
    attrs = _resolve_attrs("inertial", inertial_elem.attrib, effective_class, defaults)
    mass = float(attrs.get("mass", "0.0"))
    com_pos = np.array(_parse_floats(attrs.get("pos", "0 0 0")))

    if "fullinertia" in attrs:
        inertia = _fullinertia_to_matrix(_parse_floats(attrs["fullinertia"]))
    elif "diaginertia" in attrs:
        diag = np.array(_parse_floats(attrs["diaginertia"]))
        if "quat" in attrs:
            R = _quat_to_matrix(np.array(_parse_floats(attrs["quat"])))
            inertia = R @ np.diag(diag) @ R.T
        else:
            inertia = np.diag(diag)
    else:
        inertia = np.zeros((3, 3))

    return mass, com_pos, inertia


def _walk_worldbody(worldbody, defaults, force_limits):
    # DFS over <worldbody>.  Fixed bodies merge inertia into nearest moveable ancestor.
    # freejoint bodies are treated as fixed (floating-base DOF, not a chain joint).
    joints_out = []
    links_out = []

    def walk(body_elem, parent_childclass, T_ancestor_to_parent, cur_link):
        # T_ancestor_to_parent: 4x4 from last moveable ancestor's frame to this body's parent
        # cur_link: mutable dict for the last moveable ancestor's link, or None
        childclass = body_elem.get("childclass", parent_childclass)

        body_pos = np.array(_parse_floats(body_elem.get("pos", "0 0 0")))
        quat_str = body_elem.get("quat")
        body_rot = (
            _quat_to_matrix(np.array(_parse_floats(quat_str)))
            if quat_str is not None else np.eye(3)
        )

        T_body = np.eye(4)
        T_body[:3, :3] = body_rot
        T_body[:3, 3] = body_pos

        T_ancestor_to_body = T_ancestor_to_parent @ T_body
        R_anc = T_ancestor_to_body[:3, :3]
        p_anc = T_ancestor_to_body[:3, 3]

        joint_elems = [e for e in body_elem if e.tag == "joint"]
        freejoint_elems = [e for e in body_elem if e.tag == "freejoint"]
        inertial_elem = body_elem.find("inertial")

        has_moveable = bool(joint_elems) and not freejoint_elems
        has_freejoint = bool(freejoint_elems)

        if has_moveable:
            for joint_elem in joint_elems:
                joint_class = joint_elem.get("class", childclass)
                eff = _resolve_attrs("joint", joint_elem.attrib, joint_class, defaults)

                jtype_str = eff.get("type", "hinge")
                if jtype_str == "hinge":
                    jtype = 0
                elif jtype_str == "slide":
                    jtype = 1
                else:
                    continue  # skip ball/free joints

                jname = joint_elem.get("name", body_elem.get("name", ""))
                axis = np.array(_parse_floats(eff.get("axis", "0 0 1")))
                rng_str = eff.get("range")
                lo, hi = _parse_floats(rng_str) if rng_str else [-np.inf, np.inf]
                flo, fhi = force_limits.get(jname, (-100.0, 100.0))

                joints_out.append({
                    "name": jname,
                    "type": jtype,
                    "axis": axis.tolist(),
                    "lower_limit": lo,
                    "upper_limit": hi,
                    "max_force": fhi,
                    "max_velocity": 100.0,
                    "parent_frame_pos": p_anc.tolist(),
                    "parent_frame_rot": R_anc.tolist(),
                })

                mass, com_pos, inertia = (
                    _parse_inertial(inertial_elem, childclass, defaults)
                    if inertial_elem is not None
                    else (0.0, np.zeros(3), np.zeros((3, 3)))
                )
                new_link = {
                    "_mass": mass,
                    "_com": com_pos.copy(),
                    "_I": inertia.copy(),
                }
                links_out.append(new_link)

                # This body is now the moveable ancestor; reset T for children.
                for child_body in body_elem.findall("body"):
                    walk(child_body, childclass, np.eye(4), new_link)

                break  # only first hinge/slide per body (serial chain assumption)

        elif has_freejoint:
            for child_body in body_elem.findall("body"):
                walk(child_body, childclass, np.eye(4), None)

        else:
            # Fixed body: merge inertia into nearest moveable ancestor if there is one.
            if cur_link is not None and inertial_elem is not None:
                mass_fix, com_fix_body, I_fix_body = _parse_inertial(
                    inertial_elem, childclass, defaults
                )
                if mass_fix > 0:
                    com_fix_anc = p_anc + R_anc @ com_fix_body
                    I_fix_anc = R_anc @ I_fix_body @ R_anc.T
                    m_new, com_new, I_new = _merge_inertia(
                        cur_link["_mass"], cur_link["_com"], cur_link["_I"],
                        mass_fix, com_fix_anc, I_fix_anc,
                    )
                    cur_link["_mass"] = m_new
                    cur_link["_com"] = com_new
                    cur_link["_I"] = I_new

            # Continue descent with accumulated transform (not reset).
            for child_body in body_elem.findall("body"):
                walk(child_body, childclass, T_ancestor_to_body, cur_link)

    for top_body in worldbody.findall("body"):
        freejoint_elems = [e for e in top_body if e.tag == "freejoint"]
        joint_elems = [e for e in top_body if e.tag == "joint"]
        childclass = top_body.get("childclass", "")

        if freejoint_elems:
            # Floating-base: base pose is a runtime DOF, pass identity to children.
            for child_body in top_body.findall("body"):
                walk(child_body, childclass, np.eye(4), None)
        elif not joint_elems:
            # Fixed-base body may still have a non-identity pos/quat (e.g. UR5e base
            # has quat="0 0 0 -1").  Propagate that static transform so children get
            # correct world-frame joint positions.
            base_pos = np.array(_parse_floats(top_body.get("pos", "0 0 0")))
            quat_str = top_body.get("quat")
            base_rot = (
                _quat_to_matrix(np.array(_parse_floats(quat_str)))
                if quat_str is not None else np.eye(3)
            )
            T_base = np.eye(4)
            T_base[:3, :3] = base_rot
            T_base[:3, 3] = base_pos
            for child_body in top_body.findall("body"):
                walk(child_body, childclass, T_base, None)
        else:
            walk(top_body, childclass, np.eye(4), None)

    return joints_out, links_out


def parse_mjcf(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    defaults = _build_defaults(root)
    force_limits = _parse_actuators(root, defaults)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> element found in {filename}")

    joints, links = _walk_worldbody(worldbody, defaults, force_limits)

    n = len(joints)
    if n != len(links):
        raise RuntimeError(f"Parser internal error: {n} joints but {len(links)} links")

    return {
        "num_joints": n,
        "num_non_base_links": n,
        "joint_names":                  [j["name"]             for j in joints],
        "joint_types":                  [j["type"]             for j in joints],
        "joint_lower_limits":           [j["lower_limit"]      for j in joints],
        "joint_upper_limits":           [j["upper_limit"]      for j in joints],
        "joint_max_forces":             [j["max_force"]        for j in joints],
        "joint_max_velocities":         [j["max_velocity"]     for j in joints],
        "joint_child_link_names":       [None] * n,
        "joint_axes":                   [j["axis"]             for j in joints],
        "joint_parent_frame_positions": [j["parent_frame_pos"] for j in joints],
        "joint_parent_frame_rotations": [j["parent_frame_rot"] for j in joints],
        "link_masses":                  [l["_mass"]            for l in links],
        "link_local_inertias":          [l["_I"].tolist()      for l in links],
        "link_local_inertia_positions": [l["_com"].tolist()    for l in links],
        "link_local_inertia_rotations": [np.eye(3).tolist()    for _ in links],
        "base_pos":               None,
        "base_orn":               None,
        "base_mass":              None,
        "base_local_inertia_diag": None,
        "base_local_inertia_pos":  None,
        "base_local_inertia_orn":  None,
    }
