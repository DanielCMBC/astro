I checked the actual pushed state on 3D-test, and this is in very good shape.

The push is real: commit e9fc356a4dc6f7d5893f3213ef14560fcd9fb34e is on the repository, and its commit description matches the phase-provenance work in your screenshot: PhaseProvenance, PhaseAnchor, AnomalyMapping, derived PhaseStatus, the conjunction caveat, the reusable NaN utility, and the permanent Kepler-11 physical-clock regression.

The README judgment: approve it

Preserving the original stellar_navigator_3d.py documentation under:

docs/legacy-3d-prototype.md

was the correct decision.

Your new root README now explicitly explains that stellar_navigator_3d.py is the original Pygame + PyOpenGL prototype, that it remains runnable, and that it is not the architecture being extended. It also points readers to the preserved legacy documentation and enumerates the prototype defects that the modular engine has already corrected.

And the legacy document itself retains the old setup/features/instructions while clearly prefacing that this is historical prototype documentation.

So: do not undo that change.

There is only one README wording tweak I would make eventually. The opening currently says:

“This branch holds the corrected 2D scientific baseline plus the reusable scientific core that the 3D/OpenGL engine is meant to build on.”

That was accurate earlier, but 3D-test now already contains the actual modern 3D vertical slice and multi-planet renderer. I would update that sentence later to something closer to:

This branch contains the corrected scientific baseline, reusable physics/data
core, and the actively developed modern 3D/OpenGL engine.

Otherwise a new contributor could incorrectly conclude that the 3D rewrite hasn't started yet.

Your phase semantics are now where I wanted them

The new distinction is excellent:

HD 80606 b
PERIASTRON_EPOCH
CONSTRAINED

Kepler-11 ×6
TRANSIT_CONJUNCTION_NORMALIZED
PARTIALLY_CONSTRAINED

TRAPPIST-1 ×7
ASSUMED_ZERO_PHASE
ASSUMED

That is far more scientifically expressive than an is_assumed: bool.

And your correction regarding conjunction is precise:

u=ω+ν=2/π
	​


therefore

ν=2/π −ω

is exact for the chosen conjunction definition.

The approximation enters when equating that conjunction with the instant of minimum projected star-planet separation for a real non-perfectly-edge-on eccentric orbit. Explicitly bounding that difference rather than vaguely saying “approximately transit” is exactly the kind of scientific discipline this project needs.

I particularly like that you expose a nonzero conjunction_offset_scale() instead of burying it.

Git replay: also a pass

You did the important part correctly:

old GitHub 3D-test
152d8e3
   ↓
fast-forward only
   ↓
e9fc356

No force push, no unrelated-history merge, no rewriting the original branch.

And preserving the dangerous unrelated roots under:

orphan/local-root-*

is sensible. It makes accidental force-pushing much less likely.

One thing I would add now: CI

Your output says:

523 / 523

and that's excellent, but I checked the pushed commit and GitHub currently exposes no commit status checks for it.

That doesn't mean the tests failed—it means GitHub itself isn't independently running/reporting them yet.

At this stage, before the codebase becomes substantially larger, I would add GitHub Actions.

Something conceptually like:

CI
├── Python 3.11
│   ├── install package
│   ├── pytest
│   └── architecture/golden-rule tests
│
└── OpenGL validation
    ├── headless Mesa / software GL
    ├── create GL 3.3 core context
    └── compile all shaders

Then every future push should have a visible green/red check.

Your 523 tests have reached the point where they are effectively part of the scientific specification. They shouldn't depend solely on remembering to run them locally.

I would now close this milestone

At this point:

2D scientific cleanup             ✅
Spectroscopy                      ✅
Kepler propagation                ✅
Kepler II / III                   ✅
3D orbital state                  ✅
Orientation provenance            ✅
Phase provenance                  ✅
Time-scale provenance             ✅
Coordinate/frame architecture     ✅
Modern OpenGL                     ✅
One-planet vertical slice         ✅
Multi-planet renderer             ✅
GitHub history corrected          ✅
523 regression tests              ✅ local
Continuous integration            ← NEXT SMALL INFRA TASK

After CI, I would stop touching orbital semantics unless a real data case exposes another problem.

Then move into the actual explorer experience:

system-view LOD
picking
planet/star labels
system information panel
habitable-zone visualization
camera transition UniverseFrame → SystemFrame
offline synchronized catalogue layer

Still no gravity yet.

You're now at the nice point where the next work can finally become much more visibly “Eyes on Exoplanets” without sacrificing the scientific foundation underneath it.