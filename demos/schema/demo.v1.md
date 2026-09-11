# rigby.demo/1

One JSON object per file, `demos/registry/<id>.json`. Validated by
`demos/tools/demo_tools.py validate`; nothing outside this document is
required, and unknown keys are kept but ignored.

| key | type | meaning |
|---|---|---|
| `schema` | `"rigby.demo/1"` | the discriminator; bump it, never reinterpret it |
| `id` | string | `YYYY-MM-DD-<prompt slug>-<8 hex>`; equals the filename |
| `title` | string | one line a person reads in the viewer |
| `prompt` | string | the plain-language request the demo answers, verbatim |
| `tier` | `core` / `humanoid` / `any-robot` | which tier produced it |
| `embodiment` | string | the body: `mesh2motion-human-vrm1`, a zoo or bench `robot_id`, `gripper` |
| `kind` | see below | what the media or payload is |
| `who` | string | GitHub handle of the person who prompted it |
| `produced_at` | ISO-8601 UTC | when the run happened, not when it was registered |
| `source.commit` | string | short SHA the code ran at; `unrecorded` only for imports whose source did not keep it |
| `source.branch` | string | branch at the time |
| `source.dirty` | bool | whether the tree had uncommitted changes |
| `source.how` | string | the command that produced it, runnable from the tier directory |
| `outcome` | `{state, text}` or null | `ok` / `partial` / `failed`, and the number that says so |
| `notes` | string | anything the viewer should show under the card |
| `media[]` | `{path, bytes, sha256, role}` | repo-relative; `role` is `clip`, `still` or `comparison` |
| `payload` | `{path}` or null | a pose payload for the pose kinds; repo-relative |
| `tags[]` | strings | free |

## kind

| kind | media | payload | player |
|---|---|---|---|
| `gif`, `video`, `image` | required | none | the media itself |
| `humanoid-bones-v1` | optional still | `clip.json` from `rigby_poc` (52 named bones, local delta quaternions relative to the GLB rest pose, glTF Y-up) | bone player, phase 2 |
| `anyrobot-qpos-v1` | optional still | the viewer export from `any-robot/scripts/export_viewer.py` (`scene` + `qpos` rows, MuJoCo Z-up) | FK player, phase 2 |
| `gripper-links-v1` | optional still | a `gripper_clip_v1` recording (per-frame world-space link segments) | segment player, phase 2 |

The three pose kinds exist because their frames cannot be converted into one
another without the rig: named-bone rotations, joint-space `qpos`, and
world-space segments are three different things. The viewer dispatches on
`kind` and never guesses.

## Limits

8 MB per file, 24 MB per entry, media types gif/webm/mp4/png/jpg. Larger
recordings go on a GitHub release with the URL in `notes` and a still in
`media`.
