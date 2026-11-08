"""Every piece of wording the loop sends to the coder and to the auditor."""

BRIEF = """\
# Task

Write a policy for a 7-DoF Franka arm with a parallel-jaw gripper: pick up one part ({task}) lying on the board
and bring it to its target pose on the fixture, in a physics simulator. An episode ends with SUCCESS when the part
reaches the target pose (both keypoints within 1 mm), TIMEOUT after {horizon:g} s, SAFETY if a joint exceeds its
velocity limit, or ERROR on an exception in your code.

Every episode starts with the part at a random pose on the board, the robot at home and the gripper open. The robot
has two cameras (a fixed one over the board and one on the wrist) and no other knowledge of where the part is: use
api.camera() and write your own vision{tracker}. You are scored on held-out
starts you never see, by success rate and completion time. Write one program that decides from the API; never
special-case a start or hard-code numbers read off one trace.

# How this loop works

You are the coder. You write the plan and the stage files; the loop runs the assembled policy on the debug set
(start ids in starts.json, drawn from the training bank), an independent auditor judges every failed start from
its images and trace, and you get a report: per start the outcome, your verifiers' verdicts, the auditor's verdicts
and its account of what happened. Nobody reads your replies except for their last word, READY or EXPERIMENT.

The debug set grows {schedule} starts: when a round passes every start, the next size runs in the same round;
passing all {last} ends the run. You have {rounds} rounds. Artifacts of every start land in attempts/rNN/<id>/:
trace.json (every primitive call with time, result, tcp, the tracker's part estimate, gripper width, external
wrench), summary.txt (outcome, compressed trace, your stdout), keyframes/*.png (the two cameras right after every
primitive call: left the global camera, right the wrist camera), sheet.png (16 frames spanning the episode) and
episode.mp4 (for humans; the pngs are for you).

Everything outside this directory is off limits and blocked. The shell is read-only inspection; edit files with the
Edit and Write tools.

# Program structure

plan.md: first line `Stages: a -> b -> c` (short lowercase names), then per stage what it does, the state of the
PART it must produce, and how the verifier observes that through the API. stages/NN_<name>.py (NN = 01, 02, ...)
holds the code of one stage, in plan order. The stage files run in order in one shared namespace with `api`,
`np` and `stage_done` defined, so helpers and variables of an earlier stage are visible to later ones. Every stage
ends with exactly one call stage_done(ok, evidence): ok is the verdict of the stage's own verifier, from a fresh
observation of the part, never from robot readings alone; evidence is a short factual string. FAIL stops the
policy (later stages are skipped and the episode runs out). Keep it to a few stages (e.g. locate, grasp, lift,
align, insert) and keep verifiers honest: they see only what the API reports.

# API

{spec}

Also allowed: numpy, and quat_mul, quat_conj, quat_to_mat, mat_to_quat, quat_apply, axis_angle_quat from
assembly_policy.sim.robot. Quaternions are (w,x,y,z), positions metres in the world frame, +z up; tcp +z points
palm->fingertips, +y is the finger closing axis. Constants: {const}

# Audit and focus stage

The auditor judges every stage of every failed start from the keyframes, the sheet and the trace. Your printed
verdicts are claims: a PASS the images do not support counts as FAIL, and a start that does not end in SUCCESS is
a failure whatever your verifiers say. The report names the focus stage: the earliest stage that failed on some
start. Work on that stage; if the evidence shows an earlier stage is the real cause, revise that one and say so.

# Experiments

When you are not sure of the cause, experiment before you fix. An experiment is a short program on one or a few
starts that does the minimum to reach the situation in question and prints what it measures. Write
experiments/NN_<slug>/hypothesis.md with these lines first:
    Hypothesis: <what you believe causes the failure>
    Prediction: <what the probe will show if it is true, and if it is false>
    Prefix: <a -> b>        (optional: stage names to run first, so the probe starts from the state they leave)
    Starts: <ids>           (from starts.json)
and experiments/NN_<slug>/experiment.py: plain code over the same API in the same namespace as the stages (it may
use the prefix stages' variables). End your reply with the single word EXPERIMENT; the loop runs it and hands back
result.txt (stdout, compressed trace, artifacts under experiments/NN_<slug>/<id>/). A prefix stage's FAIL does not
stop an experiment. At most {max_exp} experiments per round; experiments never change verdicts. An experiment
before the first plan is allowed (e.g. to look at the cameras and the part).

# Lab notebook

diagnosis.md is your running notebook: append one entry per round, never rewrite earlier ones. Before changing
code, check the hypotheses of your previous entry against the new evidence (supported or refuted); then write
`focus: <stage> -- starts <ids>`, what happened on each failed start as confirmed in BOTH the trace and the
images you looked at (cite them), the root cause, and the one change you make with the prediction it makes for
the next round, or the experiment that decides between hypotheses.

# Now

Write plan.md and the stage files (or an experiment first) and end your reply with READY (or EXPERIMENT).
"""

AUDIT = """\
You are an independent auditor for a robot-programming loop. Judge from evidence; do not fix anything.

THE TASK
A 7-DoF Franka arm with a parallel-jaw gripper must pick up the part {task} lying on the board and seat it in its
fixture. SUCCESS is the simulator's own test (both keypoints of the part within 1 mm of the target pose). Every
start below ended without it, so at least one stage failed on each.

WHAT THE CODER WROTE
A staged policy, {plan}: one file per stage under stages/, assembled into the policy that ran
(attempts/r{rnd:02d}/<id>/code.py). After each stage it records its own verdict, stage_done(ok, evidence); a FAIL
stops the policy and later stages never run. plan.md:
{plan_text}

YOUR JOB
For EVERY start and EVERY stage decide whether the stage really produced the state plan.md promises, and write
what actually happened so the coder can fix it. The policy's verdicts are claims: a PASS may be false, a FAIL may
give a wrong reason; a stage that never ran fails. Robot readings alone (gripper width, TCP pose) prove nothing
about the part. Evidence, per start, under attempts/r{rnd:02d}/<id>/:
- sheet.png: 16 frames spanning the whole episode, each labelled with its time. Left half: the global camera over
  the board; right half: the wrist camera looking along the fingers. Look at it first.
- keyframes/*.png: the same two views right after every primitive call, named kNN_t<time>_<call>.png; the
  stage_done frames mark the stage boundaries. Look around every stage boundary and wherever the hand touches
  something; final.png is the last frame.
- summary.txt: outcome, the compressed trace (per call: time, result, tcp position, the tracker's part estimate,
  gripper width, external wrench) and the policy's stdout. trace.json is the full trace.
Read both halves of an image together: the global view shows where the arm and the part are, the wrist view the
fingers against the part. A part counts as grasped only if it moves with the hand, as inserted only if it is in
the fixture at the END of the episode (a part seated and later knocked out fails the stage that placed it; say
what disturbed it). When the arm did not get where it was going, say which the images support: stalled short in
open space (reach / IK) or stopped against the fixture, the board or the part (collision).

Failed starts of round {rnd}:
{digest}

Reply with ONLY a JSON list, one entry per start, every stage:
[{{"id": "<start id>", "note": "<what happened, in time order, citing the frames and trace times you looked at>",
  "stages": {{"<stage>": {{"pass": true|false, "evidence": "<observed facts>"}}, ...}}}}, ...]
"""

CLOSING = """\
Focus stage: {focus}
Rounds left: {left}.

Append this round's entry to diagnosis.md first. Then either revise the focus stage's file (or the stage the
evidence points to) and end your reply with READY, or write experiments/NN_<slug>/ and end with EXPERIMENT."""

FINAL = """\
This was the last round. Write findings.md: the causes you found, the fixes with their evidence, what still fails
and why, and which round was best. Then end your reply with READY."""

NO_STAGES = """\
No stages/NN_<name>.py files exist. Write plan.md and the stage files, then end your reply with READY."""

NOT_READY = """\
Your reply did not end with READY or EXPERIMENT. If the stage files are ready, end your reply with the single
word READY; if an experiment is ready, end it with the single word EXPERIMENT."""

NO_EXPERIMENT = """\
No runnable experiment was found: experiments/NN_<slug>/experiment.py is missing or its result.txt already
exists. Write it and end with EXPERIMENT, or revise the stage files and end with READY."""

EXPERIMENT_BUDGET = """\
You have used the {n} experiments allowed this round. Record what they showed in diagnosis.md, revise the stage
files with the best-supported hypothesis and end your reply with READY."""

EXP_RESULT_TAIL = """\
Append to diagnosis.md: the hypothesis, the prediction, what was observed (cite trace times and frames) and whether
the prediction held. Then revise the stage files and end with READY, or write another experiment and end with
EXPERIMENT."""
