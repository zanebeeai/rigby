# Pinned third-party intake fixtures

`source-packages.zip` contains three complete model descriptions and all 41
referenced meshes, plus a URDF-only Panda coupling-refusal case. Tests verify
the archive digest and every entry, then extract locally without networking.

The iiwa7 and kuka_lwr sources come from bulletphysics/bullet3 at
`63c4d67e337017f9d8b298c900e9aabdb69296e7` under its Zlib license. The SO-ARM101
source comes from TheRobotStudio/SO-ARM100 at
`7629d2ad9853d10fb903093a33ef6114099d97e5` under Apache-2.0. Original notices
and attribution are included. Rigby does not author these models.

The three admitted URDFs and every referenced mesh were compared byte-for-byte
with their exact pinned upstream URLs. `provenance.json` records those URLs,
hashes and comparison scope. Renaming the local URDF file to `robot.urdf` is the
only packaging operation for those three descriptions. The separate Panda
fixture preserves its previously documented package-path localization; its
unsupported mimic coupling is refused before mesh loading.

`python package.py extract destination` verifies and extracts offline.
`python package.py build path/to/exotic` explicitly rechecks the existing local
assets against upstream and rebuilds the archive; this acquisition command
requires networking and is never called by the tests.
