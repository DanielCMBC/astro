# Explorer C3.5 History Rewrite Approval

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Remote tip:** `f470cfd4e42d2ed6dce02039226bbd22aa43672b`  
**Revised local C3.5 commit:** `fb3d5a5`  
**Reported verification:** `910 passed, 1 skipped`  
**Decision:** **Use the trailer-removal path. Do not push yet until the shared-branch coordination check is complete.**

## Scientific status

The revised C3.5 work now addresses the previously identified blockers:

- canonical basis is `+X = East`, `+Y = North`, `+Z = away from observer`;
- astronomical position angle is converted at one boundary using `theta_math = pi/2 - Omega_PA`;
- `NodeConvention` is explicit and unspecified conventions fail closed;
- `NodeSense` distinguishes resolved node sense from modulo-180 ambiguity;
- a valued `Omega` no longer automatically means a unique ascending node;
- projected `(omega, Omega)` versus `(omega+pi, Omega-pi)` degeneracy is tested;
- coordinate epoch remains unresolved by default, so absolute instantaneous positions stay withheld;
- exact-pole tangent azimuth is treated as degenerate;
- planet-to-selected-star distance is restored only behind the full publication gates;
- 910 tests pass locally, with one deliberate pole derivative skip.

One semantic rule should remain explicit: `NodeSense.RESOLVED` must come from orbit-specific line-of-sight evidence or an equivalent observation that genuinely identifies the ascending/descending node. A generic systemic stellar radial velocity is not enough by itself.

## History decision

The remote branch still contains the published C3 commit with the unwanted `Co-Authored-By` trailer. Because the requirement is that this attribution not remain in the **active branch history**, preserving the existing public history does not satisfy the requirement.

Choose the **trailer-removal rewrite** path.

### Important safety adjustment

Do **not** create a remote backup branch pointing at the old attributed commit, because that would deliberately keep the unwanted history reachable from the repository.

Create the backup locally instead:

```bash
git branch backup/3D-test-pre-attribution-rewrite f470cfd4e42d2ed6dce02039226bbd22aa43672b
```

Optionally create a local bundle:

```bash
git bundle create ../3D-test-pre-attribution-rewrite.bundle   f470cfd4e42d2ed6dce02039226bbd22aa43672b
```

Do not commit or upload that bundle.

## Shared-branch precondition

Before rewriting `3D-test`, confirm with any collaborators that nobody has unpublished/unmerged work based on:

```text
b8b4457
f470cfd
```

GitHub cannot prove the absence of unpublished local commits on someone else's machine.

If someone does have work based there, they should preserve it before the rewrite and later rebase/cherry-pick onto the rewritten history.

## Exact rewrite sequence

After collaborator coordination is confirmed:

```bash
git fetch origin
git rev-parse origin/3D-test
```

The result must still be exactly:

```text
f470cfd4e42d2ed6dce02039226bbd22aa43672b
```

If it differs, stop.

Verify local rewritten history:

```bash
git log --format=full 5129422..HEAD
```

and ensure there are no unwanted co-author trailers.

Then use the exact-SHA lease:

```bash
git push   --force-with-lease=refs/heads/3D-test:f470cfd4e42d2ed6dce02039226bbd22aa43672b   origin HEAD:refs/heads/3D-test
```

Never use plain `--force`.

If the lease fails, stop and investigate.

## Post-rewrite checks

After the rewrite:

```bash
git fetch origin
git rev-parse origin/3D-test
```

The remote tip must equal the full rewritten C3.5 SHA.

Then inspect the active branch history:

```bash
git log --format=%B 5129422..origin/3D-test
```

and verify there are no unwanted:

```text
Co-Authored-By
Claude
Anthropic
```

trailers.

Then report:

```text
branch: 3D-test
remote SHA: <full rewritten C3.5 SHA>
CI run: <id>
CI status: green / failing
```

## What this rewrite guarantees

A branch rewrite removes the unwanted commits from the **active branch history**.

It does not guarantee instant destruction of old Git objects everywhere. Existing clones, forks, cached object storage, or direct old commit URLs may retain the old SHA for some time.

If the requirement is complete server-side erasure of an old object rather than removal from active history, that is a separate GitHub-support problem.

## Final instruction

**Choose the trailer-removal path.**

Keep the rollback backup local, confirm no collaborator has unpublished work based on the old C3 commits, verify the remote tip is still `f470cfd...`, then perform one exact-SHA `--force-with-lease` rewrite.

Afterward send the new remote SHA and CI result for the final C3.5 audit.
