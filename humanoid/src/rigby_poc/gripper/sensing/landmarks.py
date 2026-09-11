"""The model says what is there and where; geometry turns that into metres.

The perception this replaces could locate exactly one object and knew which one
by construction -- a hardcoded test for the only warm thing on a blue-grey
bench. Everything else in the room, including the bin the block had to end up
in, was invisible to every instrument, and the machine knew where the bin was
only because a manifest said so. Asked why the corner cameras did not perceive
the bin, the honest answer was that nothing had ever tried.

The first repair was worse in an interesting way: a vocabulary of colour words,
with the model choosing "orange" for the block and "green" for the bin. That
still hands the model the answer. Whoever writes the vocabulary has already
decided what the scene contains and how to find it, and a new object needs a new
word from a person.

So nothing here is told what to look for. THE MODEL LOOKS AT THE PICTURES AND
POINTS. It is given every room view at a stated size -- four corners and one
steep overhead -- and reports, for each thing the task requires, the pixel it
appears at in each view it can see it in. Rays through those picks cross in one
place, and that place is in metres.

Three things fall out of doing it this way:

WHAT COUNTS AS AN ITEM comes from the task, not from this file. "Put the block
in the bin" needs a block and a bin. A different sentence needs different
things, and none of the code below changes.

DISAGREEMENT IS MEASURED. Rays through picks that are not of the same object
cross badly, and the miss says so. A careful pick and a careless one produce the
same shape of answer and different misses, so the number is earned.

THE APPEARANCE IS LEARNED, NOT DECLARED. A model call costs money and seconds;
the control loop runs at 30 Hz. So the pixels around each pick are sampled and
kept as that item's look, and the tracker follows it between calls. Nobody wrote
down that the block is orange -- it is orange because that is what was under the
place the model pointed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: The cameras that watch the whole room. Any that can see a thing votes on it.
#:
#: `overhead` is not a fifth corner, it is a different elevation, and that is
#: the point. The four corners stand on one circle at the same height, so they
#: share a blind spot: measured with the block resting in the bin, all four
#: reported ZERO pixels of it and the overhead reported 496. A ring of cameras
#: cannot see into an open container no matter how many you add to the ring.
CORNERS = ("room", "corner_front_right", "corner_back_left",
           "corner_back_right", "overhead")
#: The size the corner views are rendered at. The model is told this number and
#: its picks are in these pixels, so it is also the resolution the whole scene
#: is understood at: a pick is only ever as precise as the pixel grid it is
#: made on, and at 480x360 one pixel at the far corner of the bench was about
#: 4 mm of world.
#:
#: EVERY DISTINCT SIZE COSTS A GL CONTEXT, and this codebase has run out of
#: them before -- five sizes in flight, a dozen abandoned contexts, and every
#: render coming back black, which reads exactly like a camera pointed at
#: nothing. So this size is SHARED: identification, tracking between calls, and
#: the corner picture sent with every decision are all one size rather than
#: three near-identical ones.
PICK_W, PICK_H = 768, 576
#: The hand camera, as the model sees it. Its own size because it is a
#: different aspect and a different job -- what is between the jaws, close up.
GRIP_W, GRIP_H = 640, 480
#: What TRACKING renders at, between model calls. Deliberately smaller than the
#: size the model points in, and deliberately a size that already exists so it
#: costs no extra GL context. The model's picks need resolution because a pick
#: is only as precise as its pixel grid; following a blob's centroid does not.
#: Measured on the same learned look: 768x576 gives 3.4 mm at 179 ms a refresh,
#: 320x240 gives 4.2 mm at 90 ms. Eight tenths of a millimetre is not worth
#: halving the rate at which the belief can be corrected.
TRACK_W, TRACK_H = 320, 240
#: Half-width of the patch sampled around a pick to learn an item's look.
_PATCH = 6
#: Fewer matching pixels than this is noise, not a thing.
_MIN_BLOB_PX = 25
#: How far a pixel's colour may sit from a learned one and still be it, in
#: normalised chroma where each coordinate runs 0..1. Loose on purpose: shading
#: moves a surface a long way in brightness and only a little in chroma, which
#: is the whole reason for dividing brightness out.
_CHROMA_TOL = 0.055
#: Rays this far from agreeing are not looking at the same thing.
AGREE_M = 0.06
#: How far a newly learned look may land from the point the model's picks put
#: the item at, before the look is judged to be of something else. Measured, a
#: look learned on the object landed within 20 mm of the anchor and one learned
#: on the bench behind it landed 220-357 mm away, so the two populations are
#: nowhere near this line from either side.
LOOK_MUST_AGREE_M = 0.10


def _chroma(image: np.ndarray):
    """Colour with brightness divided out, plus the brightness separately.

    A surface in shadow and the same surface in light sit far apart in RGB and
    close together here, which is what lets one sample of an object match the
    rest of it. The bin is lit unevenly enough across its four walls that a raw
    RGB match finds one wall and misses the others.
    """
    rgb = image[:, :, :3].astype(np.float32)
    total = np.maximum(rgb.sum(axis=2), 1.0)
    return rgb[:, :, 0] / total, rgb[:, :, 1] / total, total / 3.0


@dataclass
class Look:
    """What one item looks like, learned from where the model pointed."""

    name: str
    chroma_r: float
    chroma_g: float
    lum: float
    from_camera: str = ""
    #: How much of the sampled patch agreed with itself. A pick landing on an
    #: edge samples two things and learns neither, and this is what says so.
    coherent: float = 1.0

    def as_dict(self) -> dict:
        return {"name": self.name,
                "colour_rg": [round(self.chroma_r, 3), round(self.chroma_g, 3)],
                "brightness": round(self.lum, 1),
                "learned_from": self.from_camera,
                "patch_agreement": round(self.coherent, 2)}


def learn(image: np.ndarray, u: float, v: float, name: str,
          camera: str = "") -> Look | None:
    """What is at that pixel, as something the tracker can look for again."""
    height, width = image.shape[0], image.shape[1]
    col = int(np.clip(round(float(u)), 0, width - 1))
    row = int(np.clip(round(float(v)), 0, height - 1))
    red, green, lum = _chroma(image)
    lo_r, hi_r = max(0, row - _PATCH), min(height, row + _PATCH + 1)
    lo_c, hi_c = max(0, col - _PATCH), min(width, col + _PATCH + 1)
    patch_r = red[lo_r:hi_r, lo_c:hi_c].ravel()
    patch_g = green[lo_r:hi_r, lo_c:hi_c].ravel()
    patch_l = lum[lo_r:hi_r, lo_c:hi_c].ravel()
    if patch_r.size == 0:
        return None
    mid_r, mid_g = float(np.median(patch_r)), float(np.median(patch_g))
    spread = np.hypot(patch_r - mid_r, patch_g - mid_g)
    coherent = float((spread < _CHROMA_TOL).mean())
    return Look(name=name, chroma_r=mid_r, chroma_g=mid_g,
                lum=float(np.median(patch_l)), from_camera=camera,
                coherent=coherent)


def mask(image: np.ndarray, look: Look) -> np.ndarray:
    """Which pixels look like that."""
    red, green, lum = _chroma(image)
    near = np.hypot(red - look.chroma_r, green - look.chroma_g) < _CHROMA_TOL
    # Brightness may range widely -- that is shading -- but not collapse,
    # because in near-darkness every chroma is noise.
    return near & (lum > max(12.0, look.lum * 0.25)) & (lum < look.lum * 3.5)


def blob(image: np.ndarray, look: Look):
    """Where that look is in one image, or None if too little of it is there."""
    lit = mask(image, look)
    count = int(lit.sum())
    if count < _MIN_BLOB_PX:
        return None
    rows, cols = np.nonzero(lit)
    return {"u": float(cols.mean()), "v": float(rows.mean()),
            "u0": float(cols.min()), "u1": float(cols.max()),
            "v0": float(rows.min()), "v1": float(rows.max()),
            "pixels": count}


def _ray_through(body, camera: str, u: float, v: float,
                 width: int, height: int):
    """The world-space ray a camera casts through one of its pixels."""
    eye, orient = body.camera_pose(camera)
    focal = (height / 2.0) / float(
        np.tan(np.radians(body.camera_fovy(camera)) / 2.0))
    local = np.asarray([(u - width / 2.0) / focal,
                        -(v - height / 2.0) / focal, -1.0])
    world = orient @ local
    return np.asarray(eye), world / float(np.linalg.norm(world))


def cross(rays):
    """Where a bundle of rays comes nearest to meeting, and by how much it misses.

    Least squares for the point nearest all of them: sum (I - dd^T) p =
    sum (I - dd^T) o.

    THE MISS IS THE POINT. Two rays always cross somewhere, so a position on its
    own carries no evidence that it means anything. Four rays agreeing to three
    millimetres is evidence they are looking at the same thing; four agreeing to
    twenty centimetres is evidence they are not, and that number is the
    difference between a fix and a guess.
    """
    if len(rays) < 2:
        return None
    matrix = np.zeros((3, 3))
    vector = np.zeros(3)
    for eye, direction in rays:
        project = np.eye(3) - np.outer(direction, direction)
        matrix += project
        vector += project @ eye
    try:
        point = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return None
    miss = 0.0
    for eye, direction in rays:
        off = (point - eye) - direction * float(np.dot(point - eye, direction))
        miss = max(miss, float(np.linalg.norm(off)))
    return np.asarray(point), float(miss)


def from_picks(body, picks: dict, width: int = PICK_W, height: int = PICK_H):
    """Metres, from the pixels the model pointed at.

    `picks` maps a camera name to [u, v] in an image of the stated size. Two
    picks are the minimum; one camera pointing at something says only which
    direction it lies in, and a direction is not a position.
    """
    rays, voters = [], []
    for camera, spot in (picks or {}).items():
        if camera not in CORNERS or not spot or len(spot) < 2:
            continue
        try:
            u, v = float(spot[0]), float(spot[1])
        except (TypeError, ValueError):
            continue
        if not (0 <= u < width and 0 <= v < height):
            continue
        rays.append(_ray_through(body, camera, u, v, width, height))
        voters.append(camera)
    got = cross(rays)
    if got is None:
        return None
    point, miss = got
    return point, miss, voters


def frames(body, cameras=CORNERS, width=TRACK_W, height=TRACK_H) -> dict:
    """Every corner view, rendered once.

    Kept separate because verifying a learned look re-tracks it four times over
    the same four pictures, and rendering them per attempt was most of the cost
    of a model call.
    """
    out = {}
    for name in cameras:
        try:
            out[name] = body.view(width, height, camera=name)
        except Exception:
            continue
    return out


def track(body, look: Look, cameras=CORNERS, seen=None,
          width=TRACK_W, height=TRACK_H):
    """Where that look is now, from every camera that can still see it.

    This is what runs between model calls. It does not know what the thing is;
    it knows what the thing looked like where the model said it was, which is a
    weaker claim and a very much cheaper one.

    A LARGE OBJECT IS NOT A POINT. Each ray runs through the centroid of what
    that camera can see of the item, and for a bin viewed from four corners that
    is four different parts of it -- the near rim from one side, the far inner
    wall from the other. The rays cross near the middle rather than at it, and
    the miss reports that as the disagreement it is.
    """
    pictures = (seen if seen is not None
                else frames(body, cameras, width, height))
    rays, voters, pixels, spans = [], [], 0, []
    for name in cameras:
        image = pictures.get(name)
        if image is None:
            continue
        found = blob(image, look)
        if found is None:
            continue
        rays.append(_ray_through(body, name, found["u"], found["v"],
                                 width, height))
        voters.append(name)
        pixels += found["pixels"]
        spans.append((name, found))
    got = cross(rays)
    if got is None:
        return None
    point, miss = got
    return point, miss, voters, pixels, _size_from(body, point, spans,
                                                   width, height)


def _size_from(body, point, spans, width, height):
    """Half-extents, from how big the thing looks at the range it was found.

    The old estimate had to guess a range first, by meeting one ray with a
    bench plane whose height was declared -- so the size inherited every error
    in that assumption, and could not be computed at all for something off the
    bench. Here the range is already known: the rays crossed. Apparent size at
    a known distance is just similar triangles.

    The camera with the most pixels of it wins, because that is the one seeing
    it most squarely. The depth no single view can see is taken as the width it
    can, which is what one view always has to do.
    """
    if not spans:
        return None
    name, found = max(spans, key=lambda row: row[1]["pixels"])
    eye, _orient = body.camera_pose(name)
    distance = float(np.linalg.norm(np.asarray(point) - eye))
    focal = (height / 2.0) / float(
        np.tan(np.radians(body.camera_fovy(name)) / 2.0))
    half_wide = max((found["u1"] - found["u0"]) / 2.0 * distance / focal, 0.004)
    half_tall = max((found["v1"] - found["v0"]) / 2.0 * distance / focal, 0.004)
    return np.asarray([half_wide, half_wide, half_tall])


@dataclass
class World:
    """What the machine believes is in the room, and how sure it is of each.

    THIS IS THE IMAGINATION, and it is the thing that gets corrected. Every tick
    each item is looked for again, the measurement is compared with the standing
    belief, the gap is recorded, and the belief moves toward what was measured by
    as much as the cameras' agreement justifies. A belief that is never
    contradicted and a belief that is never checked look identical from outside,
    so the check is kept and reported even when it changes nothing.
    """

    looks: dict = field(default_factory=dict)
    belief: dict = field(default_factory=dict)
    #: Per item: how far the last measurement moved it, how far the rays missed,
    #: who voted, how many times it has been checked and when it was last seen.
    moved_by: dict = field(default_factory=dict)
    missed_by: dict = field(default_factory=dict)
    voters: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    last_seen_s: dict = field(default_factory=dict)
    #: Half-extents per item, from apparent size at the triangulated range.
    size: dict = field(default_factory=dict)
    #: Whether each item was visible at the most recent refresh.
    visible: dict = field(default_factory=dict)

    def anchor(self, body, name: str, picks: dict, now: float) -> dict:
        """Take the model's word for what is where, and learn what it looks like.

        THE LEARNED LOOK IS CHECKED BEFORE IT IS KEPT. A pick a few pixels off
        the object samples the bench behind it, and what comes back is the
        bench -- after which the tracker follows the bench forever, at 30 Hz,
        while the belief drifts a third of a metre and nothing says anything is
        wrong. Measured: a look learned on the object put the belief 2 mm from
        the block and 17 mm from the bin; one learned off the edge put it
        220-357 mm out, and the patch's own coherence did not tell the two
        apart -- it read 0.09 on a good look and 0.95 on a bad one.

        So every camera the model pointed with proposes a look, each is tried
        against the same four pictures, and the one whose tracking best agrees
        with the anchored point wins. If the best of them still disagrees badly
        then nothing was learned that finds the thing the model meant, and no
        look is kept at all: the belief stands on the picks alone and the item
        can only be corrected by looking again. Refusing to learn is a worse
        outcome than learning and a far better one than learning the bench.
        """
        placed = from_picks(body, picks)
        out: dict = {"item": name, "from": "the model's picks"}
        if placed is None:
            out["seen"] = False
            out["why_not"] = ("fewer than two cameras were given a pixel for "
                              "it, and one direction is not a position")
            return out
        point, miss, voters = placed
        self._settle(name, point, miss, voters, now)
        out.update({"seen": True, "xyz": [round(float(v), 4) for v in point],
                    "rays_missed_by_m": round(miss, 4), "seen_from": voters})

        # Learn from the pixels the model actually gave, never from the
        # triangulated point reprojected -- that samples wherever the geometry
        # landed, which on a poor pick is bench.
        pictures = frames(body, CORNERS, PICK_W, PICK_H)
        best, best_gap, tried = None, None, []
        for camera in voters:
            spot = picks[camera]
            image = pictures.get(camera)
            if image is None:
                continue
            candidate = learn(image, spot[0], spot[1], name=name, camera=camera)
            if candidate is None:
                continue
            followed = track(body, candidate, seen=pictures,
                             width=PICK_W, height=PICK_H)
            if followed is not None and followed[4] is not None:
                self.size.setdefault(name, followed[4])
            if followed is None:
                tried.append({"from": camera, "verdict": "finds nothing"})
                continue
            gap = float(np.linalg.norm(followed[0] - point))
            tried.append({"from": camera, "lands_off_by_m": round(gap, 3)})
            if best_gap is None or gap < best_gap:
                best, best_gap = candidate, gap
        out["looks_tried"] = tried
        if best is None or best_gap is None or best_gap > LOOK_MUST_AGREE_M:
            self.looks.pop(name, None)
            out["learned"] = None
            out["why_no_look"] = (
                "nothing sampled at those pixels tracks back to where the picks "
                "put it"
                + (f"; the closest was {best_gap:.3f} m off"
                   if best_gap is not None else "")
                + ". The pick probably clipped the background, so the position "
                  "is kept and the appearance is not -- this item will not be "
                  "followed between calls until it is pointed at again")
        else:
            self.looks[name] = best
            out["learned"] = best.as_dict()
            out["look_lands_off_by_m"] = round(best_gap, 4)
        return out

    def refresh(self, body, now: float) -> dict:
        """Look again for everything, compare with the belief, and correct it."""
        report: dict = {}
        pictures = frames(body)
        for name, look in self.looks.items():
            got = track(body, look, seen=pictures)
            if got is None:
                self.visible[name] = False
                report[name] = {
                    "seen_now": False,
                    "believed_xyz": self._round(self.belief.get(name)),
                    "last_seen_s_ago": (round(now - self.last_seen_s[name], 1)
                                        if name in self.last_seen_s else None),
                    "note": "not visible this instant; the belief is what it "
                            "was, and is getting older",
                }
                continue
            point, miss, voters, pixels, size = got
            if size is not None:
                self.size[name] = size
            before = self.belief.get(name)
            gap = (float(np.linalg.norm(point - np.asarray(before)))
                   if before is not None else None)
            self._settle(name, point, miss, voters, now)
            self.visible[name] = True
            report[name] = {
                "seen_now": True,
                "measured_xyz": [round(float(v), 4) for v in point],
                "believed_xyz": self._round(self.belief.get(name)),
                "measurement_disagreed_by_m": (round(gap, 4)
                                               if gap is not None else None),
                "belief_moved_by_m": round(self.moved_by.get(name, 0.0), 4),
                "rays_missed_by_m": round(miss, 4),
                "seen_from": voters,
                "pixels": pixels,
                "times_checked": self.checks.get(name, 0),
                "confidence": self.confidence(name),
            }
        return report

    def _settle(self, name, point, miss, voters, now) -> None:
        """Move the belief toward the measurement, by as much as it has earned.

        Not a jump to the new number and not an average of everything ever seen.
        The weight is how well the rays agreed: a tight crossing is taken almost
        whole, a loose one only nudges. An item measured once has nothing to
        blend with and is taken as given.
        """
        weight = float(np.clip(1.0 - miss / AGREE_M, 0.08, 1.0))
        before = self.belief.get(name)
        after = (np.asarray(point, dtype=float) if before is None
                 else np.asarray(before, dtype=float)
                 + weight * (np.asarray(point, dtype=float)
                             - np.asarray(before, dtype=float)))
        self.moved_by[name] = (0.0 if before is None
                               else float(np.linalg.norm(
                                   after - np.asarray(before, dtype=float))))
        self.belief[name] = after
        self.missed_by[name] = float(miss)
        self.voters[name] = list(voters)
        self.checks[name] = self.checks.get(name, 0) + 1
        self.last_seen_s[name] = float(now)

    def confidence(self, name: str) -> float:
        """How much the cameras agree about one item, 0 to 1. Not an opinion."""
        if name not in self.missed_by:
            return 0.0
        agree = float(np.clip(1.0 - self.missed_by[name] / 0.02, 0.0, 1.0))
        votes = min(len(self.voters.get(name, [])), 4) / 4.0
        return round(float(agree * (0.5 + 0.5 * votes)), 3)

    def at(self, name: str):
        got = self.belief.get(name)
        return None if got is None else np.asarray(got, dtype=float)

    def believed(self, name: str):
        """(where, how big, visible right now) -- what the metrics steer by.

        Returns None when the item was never placed at all, which is different
        from "not visible": a belief that exists and has gone stale is still
        the best thing to act on, and the third value says which it is.
        """
        where = self.at(name)
        if where is None:
            return None
        size = self.size.get(name)
        if size is None:
            return None
        return where, np.asarray(size, dtype=float), bool(
            self.visible.get(name, False))

    @staticmethod
    def _round(value):
        return None if value is None else [round(float(v), 4) for v in value]

    def as_dict(self) -> dict:
        return {name: {"xyz": self._round(self.belief[name]),
                       "confidence": self.confidence(name),
                       "rays_missed_by_m": round(self.missed_by.get(name, 0.0),
                                                 4),
                       "seen_from": self.voters.get(name, []),
                       "times_checked": self.checks.get(name, 0),
                       "looks_like": (self.looks[name].as_dict()
                                      if name in self.looks else None)}
                for name in self.belief}
