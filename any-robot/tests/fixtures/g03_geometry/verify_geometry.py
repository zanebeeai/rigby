"""Independent analytic verifier: standard-library XML and box geometry only.

This deliberately imports no Rigby or simulator code. It verifies the frozen
labels against source kinematics, dimensions, inertia, and opposing face contact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT=Path(__file__).resolve().parent
EPS=1e-12


def vec(raw):
    return tuple(float(x) for x in raw.split())


def add(a,b):
    return tuple(x+y for x,y in zip(a,b))


def equal(a,b):
    assert len(a)==len(b) and max(abs(x-y) for x,y in zip(a,b)) < EPS, (a,b)


def geometry(root,q):
    positions={"base":(0.,0.,0.)}
    pending=list(root.findall("joint"))
    while pending:
        advanced=False
        for joint in pending[:]:
            parent=joint.find("parent").get("link")
            if parent not in positions:
                continue
            origin=joint.find("origin")
            assert vec(origin.get("rpy"))==(0.,0.,0.), "Verifier is scoped to axis-aligned analytic fixtures"
            offset=vec(origin.get("xyz"))
            if joint.get("type")=="prismatic":
                value=q[joint.get("name")]
                limit=joint.find("limit")
                assert float(limit.get("lower"))-EPS <= value <= float(limit.get("upper"))+EPS
                offset=add(offset,tuple(value*x for x in vec(joint.find("axis").get("xyz"))))
            else:
                assert joint.get("type")=="fixed"
            positions[joint.find("child").get("link")]=add(positions[parent],offset)
            pending.remove(joint)
            advanced=True
        assert advanced,"Disconnected or cyclic joint graph"
    boxes={}
    for link in root.findall("link"):
        collision=link.find("collision")
        centre=vec(collision.find("origin").get("xyz"))
        size=vec(collision.find("geometry/box").get("size"))
        assert all(x>0 for x in size)
        world=add(positions[link.get("name")],centre)
        boxes[link.get("name")]={"centre":world,"size":size,
          "low":tuple(c-s/2 for c,s in zip(world,size)),"high":tuple(c+s/2 for c,s in zip(world,size))}
        inertial=link.find("inertial")
        equal(vec(inertial.find("origin").get("xyz")),centre)
        mass=float(inertial.find("mass").get("value"))
        inertia=inertial.find("inertia")
        x,y,z=size
        equal([float(inertia.get(key)) for key in ("ixx","iyy","izz")],
              [mass*(y*y+z*z)/12,mass*(x*x+z*z)/12,mass*(x*x+y*y)/12])
        assert mass>0 and all(float(inertia.get(key))==0 for key in ("ixy","ixz","iyz"))
    return positions,boxes


def overlap(a,b,axis):
    return min(a["high"][axis],b["high"][axis])-max(a["low"][axis],b["low"][axis])


def verify():
    freeze=json.loads((ROOT/"label-freeze.json").read_text())
    for relative,expected in freeze["files"].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    results={}
    for kind in ("rigid_tool","opposing_jaws","three_digits"):
        root=ET.parse(ROOT/kind/"robot.urdf").getroot()
        labels=json.loads((ROOT/kind/"expected.json").read_text())
        positions,boxes=geometry(root,labels["configuration"])
        equal(positions["palm"],labels["palm_origin_world_m"])
        for name,p in labels["points"].items():
            equal(add(positions[p["body"]],p["local_m"]),p["world_m"])
            if name.endswith("_distal") or name=="tool_tip":
                box=boxes[p["body"]]
                equal(p["world_m"],(box["centre"][0],box["centre"][1],box["high"][2]))
        # All authored rest collision boxes are separated, so closure evidence
        # is not contaminated by accidental assembly penetration.
        names=list(boxes)
        for i,first in enumerate(names):
            for second in names[i+1:]:
                assert not all(overlap(boxes[first],boxes[second],axis)>EPS for axis in range(3)),(first,second)
        result={"label_file_sha256":freeze["files"][f"{kind}/expected.json"],
          "analytic_points_verified":len(labels["points"]),"rest_collision_boxes_separated":True,
          "physical_grip_success_proven":False}
        if kind=="rigid_tool":
            assert sum(j.get("type")!="fixed" for j in root.findall("joint"))==2
            assert not labels["expected_grasp_capability"]
        else:
            grip=labels["gripping_region"]
            target={"low":tuple(c-s/2 for c,s in zip(grip["target_centre_world_m"],grip["target_box_size_m"])),
                    "high":tuple(c+s/2 for c,s in zip(grip["target_centre_world_m"],grip["target_box_size_m"]))}
            centre=grip["region_centre"]
            equal(add(positions[centre["body"]],centre["local_m"]),centre["world_m"])
            equal(centre["world_m"],grip["maximum_clearance_center_world_m"])
            members=labels["admissible_tip_bodies"]
            for value,key in ((0.,"maximum_aperture_m"),(.010,"half_stroke_aperture_m"),(.020,"minimum_aperture_m")):
                q={key:val for key,val in labels["configuration"].items()}
                q.update({name+"_close":value for name in members})
                _,sweep=geometry(root,q)
                left=[sweep[name]["high"][0] for name in members if sweep[name]["centre"][0]<0]
                right=[sweep[name]["low"][0] for name in members if sweep[name]["centre"][0]>0]
                assert abs(min(right)-max(left)-grip[key])<EPS
            _,closed=geometry(root,grip["contact_configuration"])
            witnesses=[]
            for name in members:
                box=closed[name]
                left=box["centre"][0]<0
                gap=target["low"][0]-box["high"][0] if left else box["low"][0]-target["high"][0]
                assert abs(gap)<EPS
                overlaps=[overlap(box,target,axis) for axis in (1,2)]
                assert min(overlaps)>0
                witnesses.append({"body":name,"normal_toward_object":[1 if left else -1,0,0],
                                  "face_gap_m":gap,"contact_rectangle_area_m2":overlaps[0]*overlaps[1]})
            assert {w["normal_toward_object"][0] for w in witnesses}=={-1,1}
            for name in set(closed)-set(members):
                assert not all(overlap(closed[name],target,axis)>EPS for axis in range(3)),name
            # The target's envelope stays inside the whole opposing support
            # envelope; the three-digit side is deliberately two patches.
            for axis,key in ((1,"allowed_center_y_interval_m"),(2,"allowed_center_z_interval_m")):
                extent_low=min(closed[name]["low"][axis] for name in members)
                extent_high=max(closed[name]["high"][axis] for name in members)
                half=grip["target_box_size_m"][axis]/2
                equal(grip[key],(extent_low+half,extent_high-half))
            result.update(kinematic_contact_witnesses=witnesses,
                          contact_q_m=.013,gripping_region_centre_verified=True)
        results[kind]=result
    return {"schema_version":"g03_independent_geometry_verification.v1",
            "independent_of_rigby_and_mujoco":True,"results":results}


if __name__=="__main__":
    print(json.dumps(verify(),indent=2,sort_keys=True))
