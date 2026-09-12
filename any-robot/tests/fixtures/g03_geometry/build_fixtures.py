"""Generate analytic fixtures and freeze labels without importing Rigby or MuJoCo."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent


def fmt(values):
    return " ".join(format(float(value), ".17g") for value in values)


def box_link(robot, name, size, mass, centre=(0, 0, 0)):
    link = ET.SubElement(robot, "link", name=name)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=fmt(centre), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=str(mass))
    x, y, z = size
    inertia = [mass * (y*y+z*z)/12, mass * (x*x+z*z)/12, mass * (x*x+y*y)/12]
    ET.SubElement(inertial, "inertia", ixx=str(inertia[0]), iyy=str(inertia[1]),
                  izz=str(inertia[2]), ixy="0", ixz="0", iyz="0")
    for kind in ("visual", "collision"):
        node = ET.SubElement(link, kind)
        ET.SubElement(node, "origin", xyz=fmt(centre), rpy="0 0 0")
        ET.SubElement(ET.SubElement(node, "geometry"), "box", size=fmt(size))
    return link


def joint(robot, name, parent, child, origin, axis=None, bounds=None):
    node = ET.SubElement(robot, "joint", name=name, type="fixed" if axis is None else "prismatic")
    ET.SubElement(node, "parent", link=parent)
    ET.SubElement(node, "child", link=child)
    ET.SubElement(node, "origin", xyz=fmt(origin), rpy="0 0 0")
    if axis is not None:
        ET.SubElement(node, "axis", xyz=fmt(axis))
        ET.SubElement(node, "limit", lower=str(bounds[0]), upper=str(bounds[1]), effort="25", velocity="0.1")
        ET.SubElement(node, "dynamics", damping="0.1", friction="0")


def platform(kind):
    robot = ET.Element("robot", name=kind)
    box_link(robot, "base", (.060,.060,.040), .8, (0,0,.020))
    box_link(robot, "slide_x", (.040,.035,.020), .25)
    box_link(robot, "slide_y", (.040,.035,.020), .20)
    box_link(robot, "palm", (.060,.050,.020), .10)
    joint(robot, "position_x", "base", "slide_x", (0,0,.070), (1,0,0), (-.05,.05))
    joint(robot, "position_y", "slide_x", "slide_y", (0,0,.040), (0,1,0), (-.05,.05))
    joint(robot, "palm_mount", "slide_y", "palm", (0,0,.040))
    return robot


def point(body, local, world, semantics, reason):
    return {"body":body, "local_m":list(local), "world_m":list(world),
            "semantics":semantics, "independent_derivation":reason}


def make(kind):
    robot = platform(kind)
    label = {"fixture_id":kind,"expected_positioning_dof":2,"position_tolerance_m":.001,
             "geometry":[],"configuration":{"position_x":0.,"position_y":0.},
             "palm_origin_world_m":[0.,0.,.150],
             "points":{"base":point("base",(0,0,0),(0,0,0),["base"],"URDF root coordinate origin"),
                       "wrist":point("palm",(0,0,0),(0,0,.150),["joint"],"0.070 + 0.040 + 0.040 m mount translations")}}
    if kind == "rigid_tool":
        box_link(robot,"tool",(.012,.012,.050),.04,(0,0,.025))
        joint(robot,"tool_mount","palm","tool",(0,0,.040))
        label.update(expected_effector_kind="tool_tip",expected_grasp_capability=False,
                     admissible_tip_bodies=["tool"])
        label["geometry"]=[{"body":"tool","box_size_m":[.012,.012,.050],"box_centre_local_m":[0.,0.,.025]}]
        label["points"]["tool_tip"]=point("tool",(0,0,.050),(0,0,.240),["tip"],
             "Centre of +Z extreme face: local 0.025 + 0.050/2; world 0.150 + 0.040 + 0.050")
        label["capability_reason"]="Single rigid tool has no movable opposing contact surfaces; tool contact is possible but self-contained grip is not established."
        return robot,label
    members = [("jaw_left",-.027,0.,.030),("jaw_right",.027,0.,.030)] if kind=="opposing_jaws" else [
        ("digit_left_low",-.027,-.012,.012),("digit_left_high",-.027,.012,.012),("digit_right",.027,0.,.036)]
    label.update(expected_effector_kind="parallel_jaw" if len(members)==2 else "multifinger",
                 expected_grasp_capability=True,admissible_tip_bodies=[name for name,*_ in members],
                 expected_opposition_groups=[[name for name,x,*_ in members if x<0],[name for name,x,*_ in members if x>0]])
    for name,x,y,width_y in members:
        box_link(robot,name,(.008,width_y,.030),.02,(0,0,.015))
        jname=name+"_close"
        joint(robot,jname,"palm",name,(x,y,.030),(-1 if x>0 else 1,0,0),(0,.020))
        label["configuration"][jname]=.010
        neutral_x=x+(-.010 if x>0 else .010)
        label["geometry"].append({"body":name,"box_size_m":[.008,width_y,.030],
                                  "box_centre_local_m":[0.,0.,.015]})
        label["points"][name+"_distal"]=point(name,(0,0,.030),(neutral_x,y,.210),["contact","tip_if_selected"],
            "Centre of +Z extreme face at half-stroke: palm Z 0.150 + joint Z 0.030 + box centre 0.015 + half-length 0.015")
    # These labels are derived from face-plane intersections, not site outputs.
    object_y=.018 if len(members)==2 else .030
    label["gripping_region"]={
        "region_centre":point("palm",(0,0,.045),(0,0,.195),["grasp_point"],
          "Mid-plane of opposing X faces; centre of their symmetric Y/Z overlapping support rectangles"),
        "target_box_size_m":[.020,object_y,.018],
        "target_centre_world_m":[0.,0.,.195],
        "contact_configuration":{**{k:0. for k in ("position_x","position_y")},
                                 **{name+"_close":.013 for name,*_ in members}},
        "minimum_aperture_m":.006,"maximum_aperture_m":.046,"half_stroke_aperture_m":.026,
        "contact_aperture_m":.020,
        "aperture_equation_m":"2 * (0.027 - q - 0.008/2)",
        "contact_q_equation_m":"0.027 - 0.008/2 - 0.020/2 = 0.013",
        "maximum_clearance_center_world_m":[0.,0.,.195],
        "allowed_center_y_interval_m":[-.006,.006] if len(members)==2 else [-.003,.003],
        "allowed_center_z_interval_m":[.189,.201],
        "all_contacts_normals_opposed":True,
        "claim":"kinematic fit and closing contact feasibility only; dynamic retention/lift and force closure are unproven"}
    origins=[p["world_m"] for k,p in label["points"].items() if k.endswith("_distal")]
    centre_x=sum(p[0] for p in origins)/len(origins)
    label["points"]["joint_origin_centroid"]=point("palm",(centre_x,0,.030),(centre_x,0,.180),["grasp_center"],
       "Arithmetic centroid of physical digit joint origins at half-stroke; distinct from the opposing-face gripping-region centre")
    return robot,label


def write_json(path, obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")


if __name__=="__main__":
    payloads={}
    for kind in ("rigid_tool","opposing_jaws","three_digits"):
        destination=ROOT/kind
        destination.mkdir(parents=True,exist_ok=True)
        robot,label=make(kind)
        ET.indent(robot,space="  ")
        urdf=ET.tostring(robot,encoding="unicode",xml_declaration=True)+"\n"
        (destination/"robot.urdf").write_text(urdf,encoding="utf-8",newline="\n")
        write_json(destination/"expected.json",label)
        for name in ("robot.urdf","expected.json"):
            payloads[f"{kind}/{name}"]=hashlib.sha256((destination/name).read_bytes()).hexdigest()
    write_json(ROOT/"invalid-cases.json",{
      "cases":[
        {"id":"inverted_position_limit","source":"rigid_tool/robot.urdf","xpath":"./joint[@name='position_x']/limit",
         "attribute":"upper","value":"-0.1","physical_defect":"lower=-0.05 exceeds upper=-0.1",
         "expected_result":"typed_refusal","acceptable_legacy_code":"unreadable_model","preferred_specific_reason":"invalid_joint_limit"},
        {"id":"negative_principal_inertia","source":"rigid_tool/robot.urdf","xpath":"./link[@name='tool']/inertial/inertia",
         "attribute":"ixx","value":"-0.000001","physical_defect":"negative principal moment of inertia",
         "expected_result":"typed_refusal","preferred_specific_reason":"degenerate_inertia"},
        {"id":"tool_has_no_gripper","source":"rigid_tool/robot.urdf","mutation":None,
         "required_capability":"GRASP","physical_defect":"No opposing articulated members exist",
         "expected_result":"request_refusal","expected_code":"unafforded_schema"}],
      "note":"Mutation recipes only; these do not add normal fixture bodies. Exact refusal details are measured separately from these preregistered physical defects."})
    payloads["invalid-cases.json"]=hashlib.sha256((ROOT/"invalid-cases.json").read_bytes()).hexdigest()
    write_json(ROOT/"label-freeze.json",{
      "schema_version":"g03_analytic_geometry_labels.v1","labels_derived_without_rigby_or_mujoco":True,
      "independence":"Analytic geometry authored before any intake comparison; no output-fitted labels.",
      "license":"CC0-1.0","source":"Original analytic fixtures authored for Rigby G03 at user request",
      "physical_execution_success_claim":False,"files":payloads})
