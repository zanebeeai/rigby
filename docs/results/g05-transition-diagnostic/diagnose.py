"""Finite diagnostic of the already-published reach/return compilation failure."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
import numpy as np
import mujoco
from rigby_core.contracts import MotionProgramV2
from rigby_core.motion.compiler import compile_motion_program
from rigby_core.motion.ownership import build_joint_series
from rigby_general.contracts import RobotAssetManifestV1
from rigby_general.pipeline import ingest_robot
from rigby_general.capabilities.intake import ingest_capability_body
from rigby_general.planner import OfflineSchemaPlanner
from rigby_general.schema.inventory import load_inventory,afforded_entries
from rigby_general.bake.enumerate import build_candidate
from rigby_general.grounding import ground

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
G01=Path('C:/Users/hocke/GitHub/rigby-g01/docs/results')
PROMPT='reach out as far as you can and then come back'

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')

def inspect(name,program,model,manifest,aliases=None):
    aliases=aliases or {}
    result={'duration_s':program.duration_s,'phases':[p.model_dump(mode='json') for p in program.phases],
       'figure_sites':[t.target for t in program.tracks],'source_joints':aliases,'extrema_violations':[]}
    try:
        trajectory=compile_motion_program(program,model,manifest,sample_hz=240)
        result['compilation']={'accepted':True,'sample_count':len(trajectory.times_s),'duration_s':float(trajectory.times_s[-1])}
    except Exception as exc:
        result['compilation']={'accepted':False,'exception':type(exc).__name__,'detail':str(exc),'details':getattr(exc,'details',{})}
    joints={j.name:j for j in manifest.morphology.joints}
    for jname,series in build_joint_series(tuple(program.tracks),program.duration_s).items():
        limits=joints[jname]
        for track in series:
            for index,segment in enumerate(track._segments):
                coefficients=np.array(segment._coefficients,float)
                derivative=np.array([k*coefficients[k] for k in range(1,6)])
                roots=np.polynomial.polynomial.polyroots(derivative)
                points=[0.,segment.duration_s]+[float(r.real) for r in roots if abs(r.imag)<1e-9 and 0<r.real<segment.duration_s]
                candidates=[(float(segment.sample(t).position),t) for t in points]
                low,lo_t=min(candidates);high,hi_t=max(candidates)
                if low < limits.minimum-1e-9 or high > limits.maximum+1e-9:
                    row={'joint':jname,'source_joint':aliases.get(jname,jname),'interval':index,
                         'interval_authored_s':[float(track.times[index]),float(track.times[index+1])],
                         'knot_values':[float(track.values[index]),float(track.values[index+1])],
                         'knot_velocities':[float(track._velocities[index]),float(track._velocities[index+1])],
                         'limits':[limits.minimum,limits.maximum],'minimum':low,'maximum':high,
                         'minimum_authored_s':float(track.times[index]+lo_t),'maximum_authored_s':float(track.times[index]+hi_t),
                         'neighbors':[{'time_s':float(track.times[k]),'value':float(track.values[k]),'velocity':float(track._velocities[k])}
                                      for k in range(max(0,index-1),min(len(track.times),index+3))]}
                    result['extrema_violations'].append(row)
    save(name+'.json',{'result':result,'program':program.model_dump(mode='json')})
    print(name,json.dumps(result['compilation']), 'overshooting_intervals=',len(result['extrema_violations']),flush=True)
    return result

index=json.loads((G01/'g01-release/index.json').read_text())
summary={'prompt':PROMPT,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'scope':'Reference compilation and analytic polynomial extrema only; no physical execution or new success claims',
         'g01_durations':[{k:row.get(k) for k in ('case','reference_duration_s','simulation_duration_s','outcome','refusal')} for row in index['cases']],
         'comparisons':{}}
with zipfile.ZipFile(G01/'g01-release/zoo_dual_arm/physical.zip') as archive:
    task=json.loads(archive.read('task.json'))
    assert task['prompt']==PROMPT
    manifest=RobotAssetManifestV1.model_validate_json(archive.read('robot.json'))
    model=mujoco.MjModel.from_xml_string(archive.read('model.xml').decode())
    program=MotionProgramV2.model_validate(task['grounded_program'])
    execution=json.loads(archive.read('execution.json'))
    summary['g01_leaf_bakes']=[{k:row.get(k) for k in ('entry_id','remove','measurements')} for row in execution['baked_primitives']]
    summary['g01_source_sha256']=json.loads(archive.read('source.json'))['package_source_sha256']
summary['comparisons']['g01_frozen_dual']=inspect('g01-frozen-dual',program,model,manifest)

source=ROOT/'any-robot/assets/general/zoo/zoo_dual_arm/robot.urdf'
inventory=load_inventory()
planner=OfflineSchemaPlanner(inventory)
for profile in ('current_legacy','current_structural'):
    capability=ingest_capability_body(source) if profile=='current_structural' else None
    robot=capability.robot if capability else ingest_robot(source,robot_id='zoo_dual_arm')
    aliases=dict(capability.manifest.joint_aliases) if capability else {}
    semantics=planner.plan(PROMPT,afforded=afforded_entries(inventory,robot.morphology))
    grounded=ground(semantics,robot.manifest,robot.finalized.model,inventory,duration_scale=1.0)
    row=inspect(profile+'-composed',grounded.program,robot.finalized.model,robot.manifest,aliases)
    row['semantic_hash']=semantics.role_normalized_hash()
    row['selected_physical_sites']=[{'site':s,'body':capability.manifest.link_aliases.get(robot.morphology.site(s).body) if capability else robot.morphology.site(s).body} for s in grounded.figure_sites]
    row['leaf_compilation']=[]
    for segment in semantics.segments:
        entry=inventory.by_binding_key(f'{segment.motion_schema.canonical_key}|{segment.figure.role.value}->{segment.ground.role.value}')
        candidate=build_candidate(entry,segment.region.remove)
        single=ground(candidate.program,robot.manifest,robot.finalized.model,inventory,duration_scale=1.0)
        leaf=inspect(profile+'-'+entry.entry_id,single.program,robot.finalized.model,robot.manifest,aliases)
        row['leaf_compilation'].append({'entry':entry.entry_id,'remove':segment.region.remove.value,**leaf})
    summary['comparisons'][profile]=row
save('summary.json',summary)
