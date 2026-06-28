"""Tests for the MJCF parser.

Serial arm models (8): FK against MuJoCo xanchor, revolute joints only.
Tree/legged models (4): parse-only validation of structure and inertia.
Franka-vs-URDF: kinematic frame comparison to the hand-validated URDF.
"""

import math
import os

import mujoco
import numpy as np
import pytest

from oscbf.core.manipulator import Manipulator
from oscbf.parsers.mjcf_parser import parse_mjcf
from oscbf.utils.urdf_parser import parse_urdf

_HERE = os.path.dirname(__file__)
_MENAGERIE = os.path.join(os.path.expanduser("~"), "mujoco_menagerie")
_M = _MENAGERIE  # shorthand

PANDA_NOHAND_XML = os.path.join(_M, "franka_emika_panda", "panda_nohand.xml")
PANDA_URDF = os.path.join(_HERE, "..", "oscbf", "assets", "franka_panda", "panda.urdf")

_SERIAL_ARMS = [
    ("franka_nohand", os.path.join(_M, "franka_emika_panda", "panda_nohand.xml")),
    ("kinova_gen3",   os.path.join(_M, "kinova_gen3",        "gen3.xml")),
    ("kuka_iiwa14",   os.path.join(_M, "kuka_iiwa_14",       "iiwa14.xml")),
    ("sawyer",        os.path.join(_M, "rethink_robotics_sawyer", "sawyer.xml")),
    ("xarm7_nohand",  os.path.join(_M, "ufactory_xarm7",     "xarm7_nohand.xml")),
    ("ur5e",          os.path.join(_M, "universal_robots_ur5e",  "ur5e.xml")),
    ("ur10e",         os.path.join(_M, "universal_robots_ur10e", "ur10e.xml")),
    ("trossen_vx300s",os.path.join(_M, "trossen_vx300s",     "vx300s.xml")),
]

# 4 tree/legged robots: parse-only (serial Manipulator FK does not apply)
_TREE_MODELS = [
    ("go2",       os.path.join(_M, "unitree_go2",          "go2.xml")),
    ("anymal_c",  os.path.join(_M, "anybotics_anymal_c",   "anymal_c.xml")),
    ("shadow_rh", os.path.join(_M, "shadow_hand",          "right_hand.xml")),
    ("unitree_h1",os.path.join(_M, "unitree_h1",           "h1.xml")),
]

_FK_TOL = 1e-5   # 10 µm; floating-point noise is ~1e-16
_N_TRIALS = 10
_RNG = np.random.default_rng(42)


def _fk_max_err(xml_path):
    # Compare Manipulator FK against mujoco xanchor for revolute joints only.
    # xanchor is configuration-independent for prismatic joints, so we skip those.
    data = parse_mjcf(xml_path)
    robot = Manipulator.from_mjcf(xml_path)
    m = mujoco.MjModel.from_xml_path(xml_path)

    jnames = data["joint_names"]
    jtypes = data["joint_types"]
    lo = np.clip(np.array(data["joint_lower_limits"], dtype=float), -2 * np.pi, 2 * np.pi)
    hi = np.clip(np.array(data["joint_upper_limits"], dtype=float), -2 * np.pi, 2 * np.pi)
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jnames]

    max_err = 0.0
    for _ in range(_N_TRIALS):
        q = _RNG.uniform(lo, hi)
        d = mujoco.MjData(m)
        for i, jid in enumerate(jids):
            if jid >= 0:
                d.qpos[m.jnt_qposadr[jid]] = q[i]
        mujoco.mj_kinematics(m, d)
        tfs = robot.joint_to_world_transforms(q)
        for i, (jid, jtype) in enumerate(zip(jids, jtypes)):
            if jid < 0 or jtype != 0:
                continue
            err = np.linalg.norm(d.xanchor[jid] - np.array(tfs[i])[:3, 3])
            if err > max_err:
                max_err = err
    return max_err


@pytest.mark.parametrize("name,xml_path", _SERIAL_ARMS, ids=[a[0] for a in _SERIAL_ARMS])
def test_serial_arm_fk(name, xml_path):
    err = _fk_max_err(xml_path)
    assert err < _FK_TOL, f"{name}: max FK error {err:.3e} m exceeds {_FK_TOL:.0e} m"


@pytest.fixture(scope="module")
def panda_data():
    return parse_mjcf(PANDA_NOHAND_XML)


def test_franka_num_joints(panda_data):
    assert panda_data["num_joints"] == 7


def test_franka_joint_names(panda_data):
    assert panda_data["joint_names"] == [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"
    ]


def test_franka_joint_types_all_revolute(panda_data):
    assert panda_data["joint_types"] == [0] * 7


def test_franka_joint_axes(panda_data):
    for axis in panda_data["joint_axes"]:
        assert np.allclose(axis, [0.0, 0.0, 1.0]), f"Unexpected axis: {axis}"


def test_franka_joint_limits(panda_data):
    default_lo, default_hi = -2.8973, 2.8973
    overrides = {
        1: (-1.7628, 1.7628),
        3: (-3.0718, -0.0698),
        5: (-0.0175, 3.7525),
    }
    for i, (lo, hi) in enumerate(
        zip(panda_data["joint_lower_limits"], panda_data["joint_upper_limits"])
    ):
        exp_lo, exp_hi = overrides.get(i, (default_lo, default_hi))
        assert abs(lo - exp_lo) < 1e-4, f"joint{i+1} lower: {lo} != {exp_lo}"
        assert abs(hi - exp_hi) < 1e-4, f"joint{i+1} upper: {hi} != {exp_hi}"


def test_franka_force_limits(panda_data):
    forces = panda_data["joint_max_forces"]
    assert len(forces) == 7
    for i in range(4):
        assert abs(forces[i] - 87.0) < 1e-6, f"joint{i+1} force: {forces[i]}"
    for i in range(4, 7):
        assert abs(forces[i] - 12.0) < 1e-6, f"joint{i+1} force: {forces[i]}"


def test_franka_link_masses(panda_data):
    expected = [4.970684, 0.646926, 3.228604, 3.587895, 1.225946, 1.666555, 7.35522e-1]
    for i, (m, e) in enumerate(zip(panda_data["link_masses"], expected)):
        assert abs(m - e) < 1e-5, f"link{i+1} mass: {m} != {e}"


def test_joint1_transform(panda_data):
    pos = panda_data["joint_parent_frame_positions"][0]
    rot = panda_data["joint_parent_frame_rotations"][0]
    assert np.allclose(pos, [0.0, 0.0, 0.333], atol=1e-6)
    assert np.allclose(rot, np.eye(3), atol=1e-6)


def test_joint2_transform(panda_data):
    pos = panda_data["joint_parent_frame_positions"][1]
    rot = panda_data["joint_parent_frame_rotations"][1]
    assert np.allclose(pos, [0.0, 0.0, 0.0], atol=1e-6)
    a = -math.pi / 2
    expected = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    assert np.allclose(rot, expected, atol=1e-6)


@pytest.fixture(scope="module")
def urdf_data():
    return parse_urdf(PANDA_URDF)


def test_joint_parent_positions_match_urdf(panda_data, urdf_data):
    mjcf_pos = np.array(panda_data["joint_parent_frame_positions"])
    urdf_pos = np.array(urdf_data["joint_parent_frame_positions"])
    assert mjcf_pos.shape == urdf_pos.shape
    assert np.max(np.abs(mjcf_pos - urdf_pos)) < 1e-3


def test_joint_parent_rotations_match_urdf(panda_data, urdf_data):
    mjcf_rot = np.array(panda_data["joint_parent_frame_rotations"])
    urdf_rot = np.array(urdf_data["joint_parent_frame_rotations"])
    assert mjcf_rot.shape == urdf_rot.shape
    assert np.max(np.abs(mjcf_rot - urdf_rot)) < 1e-3


def test_manipulator_from_mjcf_constructs():
    robot = Manipulator.from_mjcf(PANDA_NOHAND_XML)
    assert robot.num_joints == 7


def test_manipulator_from_mjcf_forward_kinematics():
    robot = Manipulator.from_mjcf(PANDA_NOHAND_XML)
    fk = robot.joint_to_world_transforms(np.zeros(7))
    assert len(fk) == 7
    assert float(np.array(fk[-1])[2, 3]) > 0.4


@pytest.mark.parametrize("name,xml_path", _TREE_MODELS, ids=[t[0] for t in _TREE_MODELS])
def test_tree_model_parses(name, xml_path):
    d = parse_mjcf(xml_path)
    assert d["num_joints"] > 0


@pytest.mark.parametrize("name,xml_path", _TREE_MODELS, ids=[t[0] for t in _TREE_MODELS])
def test_tree_model_positive_masses(name, xml_path):
    d = parse_mjcf(xml_path)
    for i, m in enumerate(d["link_masses"]):
        assert m > 0, f"{name} link{i} has zero mass"


@pytest.mark.parametrize("name,xml_path", _TREE_MODELS, ids=[t[0] for t in _TREE_MODELS])
def test_tree_model_inertia_symmetric(name, xml_path):
    d = parse_mjcf(xml_path)
    for i, I_list in enumerate(d["link_local_inertias"]):
        I = np.array(I_list)
        assert np.allclose(I, I.T, atol=1e-10), f"{name} link{i} inertia not symmetric"


@pytest.mark.parametrize("name,xml_path", _TREE_MODELS, ids=[t[0] for t in _TREE_MODELS])
def test_tree_model_inertia_rot_identity(name, xml_path):
    d = parse_mjcf(xml_path)
    for i, R_list in enumerate(d["link_local_inertia_rotations"]):
        assert np.allclose(np.array(R_list), np.eye(3), atol=1e-10), \
            f"{name} link{i} inertia_rot not identity"


def test_go2_num_joints():
    d = parse_mjcf(os.path.join(_M, "unitree_go2", "go2.xml"))
    assert d["num_joints"] == 12  # 4 legs × 3 joints


def test_anymal_c_num_joints():
    d = parse_mjcf(os.path.join(_M, "anybotics_anymal_c", "anymal_c.xml"))
    assert d["num_joints"] == 12  # 4 legs × 3 joints


def test_shadow_hand_num_joints():
    d = parse_mjcf(os.path.join(_M, "shadow_hand", "right_hand.xml"))
    assert d["num_joints"] == 24


def test_ur5e_num_joints():
    d = parse_mjcf(os.path.join(_M, "universal_robots_ur5e", "ur5e.xml"))
    assert d["num_joints"] == 6


def test_xarm7_num_joints():
    d = parse_mjcf(os.path.join(_M, "ufactory_xarm7", "xarm7_nohand.xml"))
    assert d["num_joints"] == 7


def test_trossen_has_prismatic():
    d = parse_mjcf(os.path.join(_M, "trossen_vx300s", "vx300s.xml"))
    assert 1 in d["joint_types"], "Trossen should have prismatic finger joints"
    assert d["joint_types"].count(1) == 2  # left_finger, right_finger
