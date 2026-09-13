"""Independent bounded-curve review: no IK, simulation, labels, or API calls."""
from pathlib import Path
import hashlib,json
import numpy as np
from rigby_core.contracts import InterpolationKind,MotionKeyframeV2,MotionTrackV2
from rigby_core.motion.ownership import JointTrackSeries

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
RNG=np.random.default_rng(716310)


def make(times,values,kind):
    track=MotionTrackV2(track_id='curve',target='point',owner='mechanism',interpolation=kind,
       keyframes=tuple(MotionKeyframeV2(time_s=float(t),joint_values={'coordinate':float(v)}) for t,v in zip(times,values)))
    return JointTrackSeries(track,'coordinate',float(times[-1]))


def analytic_extrema(segment):
    # Rescale actual implementation coefficients to u in [0,1], then locate
    # every real derivative root. This checks between samples independently.
    h=segment.duration_s
    c=np.array([float(coef)*h**k for k,coef in enumerate(segment._coefficients)])
    roots=np.polynomial.polynomial.polyroots(np.arange(1,6)*c[1:])
    us=[0.,1.]+[float(r.real) for r in roots if abs(r.imag)<1e-8 and 0<r.real<1]
    values=[float(np.polynomial.polynomial.polyval(u,c)) for u in us]
    return min(values),max(values)


counts={'curves':0,'intervals':0,'plateaus':0,'bound_failures':0,'bernstein_order_failures':0}
worst={'bound_error':0.,'normalized_velocity_seam_error':0.,'normalized_acceleration_seam_error':0.}
examples=[]
for case in range(1800):
    n=int(RNG.integers(3,16))
    gaps=10**RNG.uniform(-4,1.3,n-1)
    times=np.r_[0.,np.cumsum(gaps)]
    values=RNG.uniform(-3.2,3.2,n)
    if case%3==0:
        values=np.sort(values)*(-1 if case%2 else 1)
    if case%5==0:
        values[1]=values[0]
    if case%7==0:
        values[-2]=values[-1]
    series=make(times,values,InterpolationKind.BOUNDED_QUINTIC)
    counts['curves']+=1
    for i,segment in enumerate(series._segments):
        counts['intervals']+=1
        h=float(times[i+1]-times[i]);q0,q1=values[i:i+2];v0,v1=series._velocities[i:i+2]
        lo,hi=analytic_extrema(segment)
        error=max(0.,min(q0,q1)-lo,hi-max(q0,q1))
        worst['bound_error']=max(worst['bound_error'],error)
        if error>1e-10:
            counts['bound_failures']+=1
            if len(examples)<3:examples.append({'case':case,'i':i,'times':times.tolist(),'values':values.tolist(),'lo':lo,'hi':hi})
        control=np.array([q0,q0+h*v0/5,q0+2*h*v0/5,q1-2*h*v1/5,q1-h*v1/5,q1])
        increments=np.diff(control)*(1 if q1>=q0 else -1)
        if np.min(increments)<-1e-12:counts['bernstein_order_failures']+=1
        if q0==q1:
            counts['plateaus']+=1
            assert lo==hi==q0
        if i+1<len(series._segments):
            left=segment.sample(h);right=series._segments[i+1].sample(0.)
            scale=max(abs(q1-q0),abs(values[i+2]-q1),1e-12)
            local_h=min(h,series._segments[i+1].duration_s)
            worst['normalized_velocity_seam_error']=max(worst['normalized_velocity_seam_error'],abs(float(left.velocity-right.velocity))*local_h/scale)
            worst['normalized_acceleration_seam_error']=max(worst['normalized_acceleration_seam_error'],abs(float(left.acceleration-right.acceleration))*local_h**2/scale)

# Frozen scalar values from the public G01 failure, not any holdout identity.
legacy=[]
for times,values in [
    ([16.761898,18.036707,19.573017,20.932593],[2.5438834937360357,2.826298357833412,2.8492995289416974,2.785737628574271]),
    ([42.658325,44.017016,45.440577,46.29629],[2.785737628574271,2.8492995289416974,2.826298357833412,2.5438834937360357])]:
    old=make(times,values,InterpolationKind.QUINTIC);new=make(times,values,InterpolationKind.BOUNDED_QUINTIC)
    legacy.append({'old_middle_maximum':analytic_extrema(old._segments[1])[1],
                   'bounded_middle_maximum':analytic_extrema(new._segments[1])[1],
                   'same_keys':np.array_equal(old.values,new.values),'same_times':np.array_equal(old.times,new.times),
                   'old_tangents':old._velocities.tolist(),'new_tangents':new._velocities.tolist()})
result={'seed':716310,'counts':counts,'worst':worst,'examples':examples,'legacy_examples':legacy,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            ROOT/'core/src/rigby_core/contracts.py',ROOT/'core/src/rigby_core/motion/ownership.py',
            ROOT/'core/src/rigby_core/motion/compiler.py',ROOT/'any-robot/src/rigby_general/grounding/grounder.py']}}
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(result,indent=2))
