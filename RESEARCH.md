# Rigby — Founder-Grade Research Report
*Physically-grounded procedural humanoid animation from natural-language intent. July 2026.*

## 1. Executive summary & verdict

**Verdict: REWORK-THEN-GO — but only as a narrow product, and only if you drop three of the four things the pitch is proudest of.**

The current pitch — *"any prompt, any humanoid rig, no hand-keying," with an LLM director in the author's seat and a multi-view vision critic as the headline moat* — is not fundable as written. Four reasons drive the call:

1. **Every layer is commoditized, and the "distinctive piece" is the weakest one.** The physics-under-text controller is a free, open research artifact from three top labs (NVIDIA [MaskedMimic](https://research.nvidia.com/labs/par/maskedmimic/), [CLoSD](https://arxiv.org/abs/2410.03441) ICLR 2025 Spotlight, Meta [Motivo](https://arxiv.org/abs/2504.11054)). Rig-agnostic retargeting is a whole competitor's award-winning core IP ([ReConForM](https://arxiv.org/abs/2502.21207), Eurographics 2025 — authored by Kinetix). The LLM director is a weekend wrapper. And the multi-view VLM critic — the one thing "nobody ships" — is the least-proven piece: current VLMs score **near-chance on temporal motion artifacts** ([TempGlitch](https://arxiv.org/abs/2605.21443); "Time Blindness," arXiv 2505.24867), which is exactly the foot-slide/grip-drift class Rigby needs it to catch. The proprietary moat, as pitched, does not survive contact with the 2024–26 literature.

2. **The market has repeatedly declined to pay for "no hand-keying."** Every durable AI-animation business won via *augmentation* (Cascadeur, "keyframes define intent, art direction stays with the animator") or *capture* (Move.ai, Rokoko, DeepMotion). The best-funded pure text-to-motion play, [Kinetix](https://kinetix.tech/) (~$12M), repositioned to a "research lab" and licenses plumbing into Unity/Adobe. "Replace the animator" is a 2026 backlash trigger (52% of pros call generative AI net-negative, 2025 GDC survey), not a wedge.

3. **You are late to a crowded, consolidating field with a Blender-only distribution ceiling.** [Cartwheel](https://www.cartwheel.ai/) ($15.6M, ex-OpenAI/Google, 8k beta users incl. Pixar/DreamWorks, Maya/Unreal/Blender export, API coming) is the same pitch with capital and distribution. [Magic Motion AI](https://www.magicmotion.ai/) already ships the *directability wedge* — waypoints, pose keys, end-effector pins, BYO-rig, runtime engine API (live since July 2025). Autodesk (Flow Studio + MotionMaker), Unity AI, and Epic (Meshcapade) own the pipelines and bundle for free. Blender is the lowest-willingness-to-pay tier; the entire paid Blender add-on economy is ~tens of $M/yr (Superhive did ~$6.5M total in 2024).

4. **Two hard, unpriced blockers the pitch never mentions: data licensing and unit economics.** The controllers Rigby would build on train on [AMASS](https://amass.is.tue.mpg.de/license.html)/HumanML3D/SMPL, whose licenses *explicitly prohibit commercial use and commercial model training*. A shipping Rigby needs licensed data (Meshcapade for SMPL) or its own mocap capture — Kinetix spent a chunk of its raise capturing 75+ actors. That is a low-hundreds-of-$k-to-multi-$M, multi-month line item with no slot in the deck. Separately, the closed loop (RL rollout + multi-cam render + VLM × 5–15 iterations) is **minutes to tens of minutes and real dollars per clip**, against a segment anchored to $8–50/mo.

**Why REWORK-THEN-GO and not NO-GO:** there is one genuinely unoccupied space — *a Blender-native tool that takes intent + an editable constraint graph and hands the animator back fully editable F-curves on their own rig, with a physically-grounded first draft.* NVIDIA's stack is a researcher SDK; Cascadeur is assisted-keyframing not text-first; Cartwheel is browser/cloud and not Blender-native; Kinetix drifted to emotes/video. That product gap is real. It is an **integration-and-UX moat, not a technology moat** — and it is only defensible if you stop pitching the physics and the critic as the edge, and stop promising to replace the animator.

---

## 2. Product critique & recommended reworks

**What's genuinely strong:**
- The **persistent, editable constraint graph** is the correct abstraction. It matches how riggers/TDs actually think (pinned effectors, contact frames, joint limits) and is a better artifact than a baked action because it survives re-rolls. This is Rigby's best idea.
- **Physics-under-IK to enforce contacts by construction** is the right engineering seam against the documented failure of pure diffusion (foot-skate, float, drifting grips).
- The **prompt front door genuinely lowers the barrier** for the forgiving/volume tier (previz, background, blocking).

**What's broken:**
- **A prompt is a lossy spec for the two things animators are paid to author: timing and spacing.** "Vault the crate" says nothing about the ease-out, the 2-frame hold at apex, or the follow-through overlap — and there is no described surface where a human dials a curve. That granularity *is* the craft.
- **The iteration granularity is fatal to the core loop.** An animator's edit is surgical ("elbow up 3° on frame 24"). Rigby re-plans and re-bakes the whole shot — the "I liked the last one except the hand" problem, made worse by a temporal axis that destroys the timing you already approved. Nothing supports non-destructive, region-scoped edits.
- **The critic risks being an opaque, slow, nondeterministic slot machine** — and the evidence says it can't reliably see the artifacts it's built to catch. Worse, since the solver *already* computes foot contacts, COM, and constraint residuals, **numeric telemetry detects foot-slide far more reliably than any VLM**. The vision loop's headline job is better done by the solver's own signals.
- **"No hand-keying" is control-removal for the exact buyer with the money,** and — separately — a copyright liability: the US Copyright Office (Jan 2025) holds that prompt-only output is not copyrightable. A studio deliverable that is pure prompt→animation may be un-ownable.

**The reshaped v1 I'd actually bet on (sharpest wedge):**

> **A Blender-native "physics-backed first-draft + QA" tool for prop/contact-constrained action motion, that hands back fully editable F-curves.**

Concretely:
1. **Flip autopilot → power-steering.** Deliverable is native, layered, editable channels (F-curves, IK/FK, NLA tracks) — *never* a baked-only action. Borrow Cascadeur's "unbaking": sparse editable keys + labeled interpolation so the animator owns the graph the moment generation ends.
2. **Wedge on the one thing pure-generative rivals can't do: contact/prop handling.** "Weapon & prop handling" for shooter/action AA and indie — sprint, vault, reload with both hands welded to the grip. It's literally Rigby's own hero example. *(Caveat: Cascadeur 2026.1 now ships prop constraints that survive the physics solve — animator-driven, not prompt-driven, but the primitive is no longer unique. The "no one guarantees persistent contact" claim is now false.)*
3. **Surface the constraint graph as a visible, hand-editable panel.** LLM proposes; human edits; the solve is deterministic given the graph. Fixes the control problem and the reproducibility problem at once.
4. **Demote the vision critic to a tie-breaker.** Use deterministic residuals (slide velocity, contact state, COM, constraint error) as the convergence oracle. Invoke the VLM only for what it's provably competent at — *gross intent/semantic mismatch* ("did it vault or step over?") and silhouette. Build it last.
5. **Freeze every generation to a versioned deterministic artifact (plan + graph + seed)** so re-opening reproduces the shot bit-for-bit. Non-negotiable for any lead to allow it in a pipeline.
6. **Make physics optional per-channel** and expose a timing/style dial that can *violate* physics (squash/stretch, snappy anticipation) — because physics grounding is a liability for stylized work.
7. **Narrow object interaction to a curated grip/socket library** (weapon sockets, handles, ladder rungs), not general HOI. General dexterous hand-object interaction is per-skill research ([InterMimic](https://arxiv.org/abs/2502.20390), PhysHOI).

---

## 3. ICP & beachhead

**Segments, ranked:**

| Segment | Priority | Why |
|---|---|---|
| **Indie 3D game devs (Blender-native, action/RPG)** | **PRIMARY** | Rigby already lives in their tool; animation is their most-hated bottleneck; contact/prop motion maps exactly to their needs; WTP proven ($8–50/mo). |
| Previz / virtual production | Secondary (expansion #1) | Best *conceptual* fit, but Unreal/Maya-centric and small buyer count. Park until engine-agnostic. VP market ~$2.1–2.9B, 20%+ CAGR. |
| 3D-VTuber content, archviz, mograph, education | Secondary | Reachable and cheap, but low individual WTP or core use is real-time (VTuber mismatch). Education = funnel, not revenue. |
| AA/AAA, VFX | Avoid (for now) | Maya/MotionBuilder pipelines; cloud-LLM procurement blockers; a Blender add-on doesn't fit. |
| Robotics-sim / synthetic data | **Avoid — trap** | Wrong tool, fidelity, and buyer. NVIDIA Isaac/GR00T owns it; needs sim-accurate dynamics + USD/URDF + ML-engineer GTM. Chasing the funding smell = a different company. |

**The one beachhead:**

> **Solo-to-small-team indie developers building 3D action/adventure/RPG games in Blender**, who need believable, directable humanoid *action* — combat, traversal, prop/weapon interaction — they can't hand-key or afford to mocap.

**Justification:** (1) zero pipeline-switching cost — Rigby ships where they live (Superhive, itch, gamedev Discords), near-zero CAC. Chasing Maya AAA is the single biggest strategic error available. (2) Their acute pain vs. unaffordable alternatives ($80–120k/yr animator; $5–25k/day mocap). (3) The constraint-graph/contact story lands precisely where diffusion (MDM, HY-Motion) and single-shot AI mocap (DeepMotion, Move.ai) visibly fail. (4) It's a wedge: win contact-correct action clips on the dominant rigs (Mixamo, UE Mannequin, Rigify), then expand along the Blender axis, and only later go engine/Maya-native for the higher-value previz/VP market.

**Spearpoint:** one or two rig standards, and the motions where competitors are weakest — melee/traversal with weapon-and-environment contact. Win that demo unarguably first.

**The load-bearing assumption:** that Rigby's contact fidelity is *materially* better in a head-to-head vs. Magic Motion AI and diffusion-mocap incumbents. If it isn't demonstrable, this segment is a red ocean — validate before spending on GTM.

---

## 4. Competitive landscape

**Direct (same job, shipping):**
- **[Cartwheel](https://www.cartwheel.ai/)** — the one to beat. $15.6M, ex-OpenAI/Google, purpose-built *Large Motion Model* (not an LLM), production-ready editable rigs, Maya/Unreal/Blender export, 8k beta users (Pixar, DreamWorks, Roblox, Take-Two), API on roadmap. Browser-native, *not* Blender-locked. **Rigby needs a reason to exist next to it, and "we're the Blender one" isn't it.**
- **[Magic Motion AI](https://www.magicmotion.ai/)** — closest to Rigby's *directed-intent* pitch. Waypoints, paths, pose keys, end-effector pins, BYO-rig retargeting, FBX/GLB/BVH export, **runtime in-engine API since July 2025**. Critically, it's a thin product layer over open models (NVIDIA Kimodo, Tencent HY-Motion). Proves the wedge is occupiable — and occupied — by wrapping free weights.
- **[DeepMotion SayMotion](https://www.deepmotion.com/saymotion)** ($15–300/mo), **[Motorica](https://motorica.com/)** (€5M, AAA production), **Tencent HY-Motion 1.0** (open-weight, RLHF-trained to penalize foot-slide, Blender-exportable — the real in-category Chinese-lab threat), **[Cascadeur](https://cascadeur.com/)** (60k+ users, physics + now prop constraints through the solve, deliberately animator-in-control).

**Adjacent / incumbent (own distribution):**
- **Epic/Unreal** (Motion Matching in Fortnite; acquired Meshcapade), **Unity AI Generators** (first-party text-to-animation + embedded Kinetix), **Autodesk** (Flow Studio ex-Wonder Dynamics; MotionMaker in Maya 2026.1; acquired RADiCAL's core tech). All three ship native AI motion and can bundle for free. Mocap (Move.ai, Rokoko — now with first-party text-to-motion — Plask) is the "why not just capture it?" substitute.

**Academic / OSS commoditization (severe):**
- Physics+text+closed-loop core: **[MaskedMimic](https://research.nvidia.com/labs/par/maskedmimic/)**, **[CLoSD](https://arxiv.org/abs/2410.03441)**, **[ProtoMotions](https://github.com/NVlabs/ProtoMotions)** (bundles retargeting), **Meta [Motivo](https://arxiv.org/abs/2504.11054)** (all free/open). Kinematic baseline reset again in 2025: **MotionMillion** (2M sequences, zero-shot compositional) and **OmniMotion-X** (whole-body incl. hands). The VLM-critic-render-refine *pattern* is prior art (SceneCraft, LL3M, Cutscene Agent 2026).

**Defensible whitespace (honest):** No product combines (1) editable constraint graph + (2) rig-agnostic retarget + (3) physics/IK contact enforcement + (4) intent-driven authoring **natively in Blender, handing back editable curves.** That intersection is real. But *every individual axis is shallow and free*, the ML core is being open-sourced by the labs Rigby cites, and distribution belongs to Epic/Unity/Autodesk. **The moat is integration, UX, reliability, and a data flywheel — not algorithms.** The window before an incumbent bolts an LLM front-end onto an existing physics engine is ~12–24 months.

**Correction to the internal scan:** one workstream misattributed Meta's flagship humanoid-control paper (Motivo/FB-CPR) to NVIDIA, and overstated the "persistent contact through physics" whitespace (Cascadeur 2026.1 now ships it). The competitive picture is *more* crowded on Rigby's supposed wedge than first drafted.

---

## 5. Technical foundations & feasibility

**The stack, layer by layer, feasible vs. not:**

| Layer | Reality |
|---|---|
| **LLM director** (prompt → plan + constraint graph) | **Easy, not IP.** Commodity wrapper over frontier APIs. This is not the hard part. |
| **Constraint graph → IK targets** | Feasible; maps cleanly onto MaskedMimic-style masked/partial effector targets. |
| **Physics/balance sim** | **NOT real "in Blender."** Blender is Bullet-only (passive ragdoll, no balance/locomotion controller). Requires an external RL policy (PHC/MaskedMimic/CLoSD) in **Isaac Lab or MuJoCo** (note: Isaac Gym is EOL), baked to a Blender action. This is GPU-backed cloud SaaS, not an add-on. |
| **IK over physics** | The two objectives fight *only* in the naive design (IK on a finished physics pass). SOTA avoids it by encoding grips as masked targets the controller plans around. Solvable design choice, not an inherent wall. |
| **Rig-agnostic retarget** | Hard and lossy across morphologies (ReConForM). "Any conforming rig" realistically means "rigs exposing a mappable humanoid core — Rigify/Mixamo/UE/VRM — with manual fallback," not truly arbitrary. **Buy/partner (Kinetix/Meshcapade), don't build.** |
| **Vision critic** | **Weakest link.** VLMs are near-chance on temporal artifacts (TempGlitch, Time Blindness); the field's actual motion critic (MotionCritic, ICLR 2025) operates on *kinematic joint data*, not renders. Multi-view rendering also multiplies wall-clock, and the egocentric camera rarely frames the character's own feet/silhouette. |

**Infeasible-today as pitched:** "any creative prompt" is the out-of-distribution regime where text-driven physics controllers fail unpredictably (MIND 2026: "nearly random generation"; [SafeFlow](https://arxiv.org/abs/2603.23983): the SOTA response to OOD prompts is to *reject* them — directly contradicting Rigby's core promise). Reliable general HOI is unsolved.

**Hardest problems, in order:** (1) commercial motion-data rights + retraining (see §6); (2) reliable *temporal* motion critique; (3) OOD prompt reliability; (4) wall-clock latency of the loop.

**De-risked MVP scope:** bounded, in-distribution vocabulary (locomotion + weapon-carry + vault) on **one or two rigs** (Mixamo / UE5 Mannequin). Sequence the build: **(1)** prove the solver makes good motion from *fixed structured intent*, one rig, no LLM, no loop; **(2)** add the LLM director; **(3)** add the vision loop *last*, and only if deterministic residuals prove insufficient. The flashiest piece is the least load-bearing — build it last.

**Net feasibility:** buildable as a narrow research demo on bounded scope; the general "any prompt / any rig / reliable" system is 2–4 years of risk concentrated in the *solver and data rights*, not the LLM. Drop the general framing until those seams are de-risked.

---

## 6. Business model, market size, GTM & moat

**Market sizing (honest):**
- **TAM** (3D-animation software) ~$16–28B — but dominated by Autodesk/Adobe seats; not addressable. The relevant AI-animation subsegment is small: incumbent raises are $5–17M (Move.ai, Kinetix, Motorica, Cartwheel), implying a paid market in the low hundreds of $M revenue globally.
- **SAM** (Blender add-ons): the *entire* paid Blender add-on economy is plausibly tens of $M/yr. Superhive did ~$6.5M total in 2024; category leader Auto-Rig Pro sells at ~$40 one-time. **Honest ceiling for a pure add-on: low-single-digit $M ARR.**
- **SOM** (3yr, pure add-on): ~$0.5–3M ARR. To exceed it, monetize server-side compute (credits/API) and expand to Unreal/Unity where WTP is 5–10× higher. **Treat the Blender listing as distribution and funnel, not the business.**

**Business model & pricing:** Ship the add-on free/cheap (it must be GPL anyway — see moat). Core meter = **credits on compute** (DeepMotion's "1 credit = 1 second" is the proven reference). Free tier for virality; Pro ~$20–40/mo bundling a credit allowance with overage billed as credits; studio/enterprise with per-seat + credit pools + **BYO-API-key** (collapses your worst COGS line).

**Unit-economics reality:** all-in marginal cost per generation ≈ **$1–5 today** (vision-LLM calls carrying multi-frame renders + GPU physics-sim/render × 3–5 iterations; reasoning tokens can inflate 3–10×). A flat $20–30/mo sub is destroyed by one heavy user. **Margin levers are non-negotiable:** cheap models for the critic loop (Gemini Flash-tier ~$0.50/$3, Claude Haiku-tier ~$1/$5), reserve frontier models (Opus-tier ~$5/$25) for hard cases, cap iterations, cache aggressively, BYO-key. Dollar cost is manageable; **wall-clock (minutes/clip) is the real threat to the "faster than hand-keying" promise.**

**GTM:** community-led and Blender-native. The demo *is* the marketing (prompt→"sprint with the rifle, then vault" is inherently shareable). Superhive/Gumroad listing, Blender Artists / r/blender / Discords / BlenderCon, advertise Rigify + Auto-Rig Pro compatibility. Free GPL client + free generation tier as funnel; convert on credits. One mid-size design-partner studio for a credible case study.

**Moat — be blunt:**
- **No technology moat.** LLM director = wrapper; physics core = open (NVIDIA/Meta/academia); retargeting = a competitor's published IP; critic pattern = prior art *and* the novel part demonstrably fails on motion.
- **GPL kills the code moat.** Any Blender add-on importing `bpy` is a derivative work; Superhive permits only GPL/MIT; buyers get source and may redistribute. The client cannot be secret.
- **What's actually defensible, in order:** (a) a **data flywheel** — log every generation + accept/reject/edit + constraint-graph signal to fine-tune director and critic (the only compounding moat; instrument from day one); (b) the **server-side solver + eval harness** behind an API (GPL and cloning can't touch it); (c) a **proprietary, commercially-cleared motion dataset**; (d) rig-agnostic retargeting robustness; (e) brand/community/distribution.

**The unpriced blocker:** the open controllers train on AMASS/HumanML3D/SMPL — **non-commercial licenses that explicitly forbid commercial use *and* commercial training on the data.** A shipping Rigby must license SMPL (Meshcapade), license commercial motion data (Meshcapade/Rokoko), or capture its own (Kinetix's route: 75+ actors, hundreds of hours). Realistic cost: **low-hundreds-of-$k to multi-$M + months + a retraining pipeline.** This is arguably a bigger 12-month blocker than the research risk — it's a hard legal wall with a known price tag — and the deck has no line for it.

---

## 7. Top risks & open questions

1. **Platform absorption.** The solver becomes a free feature of Unreal/Unity/NVIDIA before you monetize. You'd sell what the platform gives away.
2. **Cartwheel already there, better funded, not Blender-locked.** What can Rigby do that Cartwheel can't add as a feature?
3. **Data-licensing / retraining cost + timeline** (§6) — the most concrete near-term kill risk.
4. **Vision critic doesn't work for its headline job.** Open research question with published evidence against it. Treat "VLM catches foot-slide" as an *unvalidated bet*, not a given.
5. **Loop wall-clock** undercuts the core "faster than hand-keying" promise even if quality holds; nondeterminism is a pipeline liability.
6. **Cloud-LLM-in-the-loop is a studio procurement blocker** (proprietary rigs/scripts to a cloud director). Local/on-prem is table stakes for pros.
7. **Output copyrightability** — prompt-only output isn't copyrightable (US Copyright Office, Jan 2025); studios need ownable deliverables → forces meaningful human editing into the product.
8. **"Dies in production" pattern** — AI animation redistributes work into cleanup; a baked, non-curve-editable 85%-there action is the worst ergonomics (fixing someone else's keys with no curves to grab).

**Open questions to answer before raising:** Is an auto-critic + Blender-UX layer a *company* when the hard tech underneath is free and a funded competitor already ships the directable product? Which commoditized controller do you sit on — and is its license clean? Can human-in-the-loop editing clear the copyright bar?

---

## 8. Recommended next 30/60/90 days

Cheapest experiments that kill or validate fastest, ordered by kill-power-per-dollar.

**Days 0–30 — Kill the two cheapest kill-shots first.**
- **Data-rights memo (1 week, ~$0).** Get quotes/terms from Meshcapade (SMPL commercial) and Rokoko/Meshcapade (commercial motion data). If the number is prohibitive, that reshapes or ends the plan before any code. **This is the single highest-leverage 30-day action.**
- **Critic-reliability spike (2 weeks).** On rendered clips with *known* injected foot-slide/grip-drift, measure whether a frontier VLM localizes them better than the solver's own numeric residuals. Prediction from the literature: it won't. If confirmed, demote the critic in the pitch immediately — don't let it be the headline moat.
- **Head-to-head vs. Magic Motion AI & Cartwheel (1 week).** Run Rigby's hero prompt (weapon-carry + vault) through the shipping competitors. Is Rigby's contact fidelity *visibly, unarguably* better? If not, the beachhead is a red ocean.

**Days 30–60 — Validate demand and control model.**
- **Bounded solver demo, no LLM, one rig** (Mixamo/UE Mannequin): fixed structured intent → physically-grounded locomotion + weapon-carry + vault, baked to editable F-curves. Prove the *asset* is what animators want.
- **20 design-partner interviews** in the indie beachhead: show the demo, test WTP at $20–40/mo, and test the reframed "editable-curves first draft" vs. "no hand-keying." Confirm they'll pay for *control + contact correctness*, not automation.

**Days 60–90 — Prove the loop earns its cost, or cut it.**
- Add the LLM director + deterministic-residual convergence oracle. Measure **median iterations, wall-clock, and $/clip to an acceptable result.** Set kill thresholds *now* (e.g., >3 min or >$3/clip median = the interactive product thesis fails → pivot to batch/offline "motion factory").
- Ship a rough Blender add-on that round-trips into the graph editor and NLA cleanly. The round-trip quality *is* the moat test.

**One-line summary:** The tech vision is directionally right and the market is real, but Rigby is aimed at the wrong seat, leading with its weakest moat, and silent on its two biggest blockers. Flip it to a Blender-native, curve-returning, contact-correct first-draft tool for indie action games; buy the retargeting; demote the critic; price on metered compute; and de-risk data licensing before you write the solver. Do that, and there's a fundable wedge. Ship the pitch as written, and you're reimplementing NVLabs papers into a red ocean behind Cartwheel.
