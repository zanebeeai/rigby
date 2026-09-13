"""Independent scalar reproduction: NumPy only, no Rigby, IK, or simulator.

Input is four neighboring in-limit joint keyframes for each failing interval.
The middle two define the interval; outside neighbors define its central slopes.
"""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent


def extrema(values, duration, slopes):
    p0,p1=values
    v0,v1=[v*duration for v in slopes]
    delta=p1-p0
    # Quintic on normalized u in [0,1], with zero endpoint acceleration.
    coefficients=np.array([p0,v0,0.,10*delta-6*v0-4*v1,
                           -15*delta+8*v0+7*v1,6*delta-3*v0-3*v1])
    derivative=np.arange(1,6)*coefficients[1:]
    roots=np.polynomial.polynomial.polyroots(derivative)
    phases=[0.,1.]+[float(root.real) for root in roots if abs(root.imag)<1e-9 and 0<root.real<1]
    points=[(float(np.polynomial.polynomial.polyval(u,coefficients)),u) for u in phases]
    return min(points),max(points)


def reproduce():
    cases=json.loads((ROOT/'offending-keyframes.json').read_text())['cases']
    results=[]
    for case in cases:
        keys=case['neighboring_keyframes']
        t=[key['authored_time_s'] for key in keys]
        q=[key['position_rad'] for key in keys]
        slopes=((q[2]-q[0])/(t[2]-t[0]),(q[3]-q[1])/(t[3]-t[1]))
        duration=t[2]-t[1]
        minimum,maximum=extrema(q[1:3],duration,slopes)
        lo,hi=case['joint_limits_rad']
        assert all(lo<=value<=hi for value in q), 'Every IK key is individually legal'
        assert maximum[0]>hi, 'The generated polynomial must reproduce the violation'
        assert abs(maximum[0]-case['expected_maximum_rad'])<1e-12
        _,stopped_max=extrema(q[1:3],duration,(0.,0.))
        assert stopped_max[0]<=hi+1e-12
        results.append({'case':case['case'], 'middle_interval_duration_s':duration,
          'central_slopes_rad_s':slopes,'maximum_rad':maximum[0],
          'maximum_authored_time_s':t[1]+duration*maximum[1],
          'overshoot_rad':maximum[0]-hi,'all_input_keys_legal':True,
          'zero_slope_diagnostic_maximum_rad':stopped_max[0],
          'zero_slope_diagnostic_scope':'Shows fixed keys admit a bounded scalar interpolant; speed/acceleration/task geometry still require validation'})
    return results


if __name__=='__main__':
    print(json.dumps(reproduce(),indent=2))
