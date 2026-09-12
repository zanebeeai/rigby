# Independent analytic geometry fixtures for G03

These three original CC0 fixtures and labels were authored from explicit box dimensions before comparing against Rigby's intake output. They use no zoo expected-site files, sealed manifests, third-party assets, learned labels, or simulator-generated expected coordinates. `build_fixtures.py` imports only the Python standard library and writes the URDFs, expected labels, invalid-case recipes, and a SHA-256 label-freeze manifest.

`verify_geometry.py` independently parses those URDFs and computes forward kinematics, box bounds, uniform-box inertia, named face centres, aperture extrema, and opposing contact rectangles. It imports neither Rigby nor MuJoCo. Run it directly with Python; any altered frozen label or inconsistent geometric claim fails an assertion. The fixed point-label tolerance is **1 mm**. Analytic self-checks use 1e-12 m numerical tolerance.

| Fixture | Mechanism | Independent geometry |
| --- | --- | --- |
| `rigid_tool` | Two prismatic positioning axes, one rigid distal box | Tool box 12 × 12 × 50 mm; distal face centre is tool-local (0, 0, 50) mm, world (0, 0, 240) mm. No opposing articulated surfaces establish grasp capability. |
| `opposing_jaws` | Two positioning axes plus two inward sliding jaws | Each jaw is 8 × 30 × 30 mm. Opening ranges 46 to 6 mm. A 20 × 18 × 18 mm target contacts both jaw faces at 13 mm closing displacement, centred at world (0, 0, 195) mm. |
| `three_digits` | Two positioning axes; two fingers oppose one broader finger | Left digits are 8 × 12 × 30 mm at Y = ±12 mm; right digit is 8 × 36 × 30 mm at Y = 0. A 20 × 30 × 18 mm target simultaneously contacts all three at 13 mm closing displacement, centred at world (0, 0, 195) mm. |

All source dimensions are metres, masses kilograms, prismatic joint displacements metres, inertia kg·m², and effort limits newtons. Every fixture has finite mass, exact uniform-box diagonal inertia, declared bounded prismatic ranges, and separated rest collision boxes. The shared palm origin is world Z = 70 + 40 + 40 = 150 mm. The label inspection configuration sets positioning joints to zero and each closing joint to 10 mm (half-stroke); `expected.json` declares this explicitly. It is an independently specified configuration and does not assume the runtime's chosen neutral pose.

The labels distinguish two physically different points:

- `grasp_center`: the arithmetic centroid of the digit joint origins at half-stroke. This is at palm-local Z = 30 mm. On the asymmetric three-digit layout its X coordinate is -17/3 mm.
- `grasp_point`: the maximal-clearance centre of the region where the stated finite box fits and has opposing contact support. This is palm-local (0, 0, 45) mm. It is not generally the midpoint of an arbitrary single ray between digit surfaces.

Every digit's distal face centre is independently labelled as a contact site. A selected TIP site may belong to any listed physically admissible digit; the fixture does not assign primary physical identity using alphabetical names. A production invariance test must require the same geometric selection or an explicitly represented symmetry equivalence class across renaming.

The contact proof establishes a kinematically feasible configuration, zero face gaps, positive overlap areas, and opposing X normals. It does **not** establish force closure, frictional retention, dynamic lifting, robustness, controller success, or hardware feasibility. For the three-digit fixture, the two left contact patches are separated by a gap; a box spanning both patches remains a valid geometric grasp candidate even though a ray through its centre passes through that gap.

`invalid-cases.json` declares two source mutation recipes and one missing-capability request: inverted position limits must be refused, negative inertia must be refused, and rigid-tool transport requiring a grasping effector must be refused. These recipes add no normal fixture bodies. Their expected physical defects were frozen before the intake comparison. Broad legacy error codes may lack the specific reason desired by G03; the independent labels should not be changed to match an incorrect acceptance.

Generated artifacts may be reconstructed with `build_fixtures.py`; regeneration must produce the same payload hashes. After committing these fixtures, any intentional label revision should be separately reviewed and recorded as a new freeze rather than silently fitted to implementation output. Current intake comparison results belong under the ignored `any-robot/results/g03-geometry-fixture/`, not in the expected labels.
