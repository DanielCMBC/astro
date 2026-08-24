# C3.5 Rewrite Gate — What Can and Cannot Be Confirmed Remotely

**Repository:** `DanielCMBC/astro`  
**Branch:** `3D-test`  
**Expected remote lease target:** `f470cfd4e42d2ed6dce02039226bbd22aa43672b`  
**Prepared local tip:** `d7082f8`  
**Reported verification:** `912 passed, 1 skipped`

## Remote facts I can confirm

The public `3D-test` branch is still at:

```text
f470cfd4e42d2ed6dce02039226bbd22aa43672b
```

The repository currently has no open pull requests.

Those facts reduce the chance that visible remote work would be overwritten.

## What I cannot verify

I cannot prove that another collaborator has no unpublished local commits based on:

```text
b8b4457
f470cfd
```

GitHub cannot expose work that exists only on someone else's machine.

Therefore the shared-branch precondition must be satisfied by human coordination:

- if you are the only person working on `3D-test`, that is sufficient;
- otherwise confirm with the other contributors that they have preserved any unpublished work before the rewrite.

## Why I cannot execute the push myself

The rewritten commits:

```text
d9224cf
9ccdd85
d7082f8
```

exist only in your local repository at `E:\astro-main`.

The GitHub-side tools can move a branch only to a commit object that already exists on GitHub. They cannot upload your local Git objects or execute a shell command in your local checkout.

So the exact `git push --force-with-lease` must be run from your local repository.

## Approval condition

If you confirm either:

```text
I am the only person with work based on the current 3D-test history
```

or:

```text
all collaborators have preserved/rebased their unpublished work
```

then the rewrite is approved.

Run:

```bash
git fetch origin
git rev-parse origin/3D-test
```

It must still print:

```text
f470cfd4e42d2ed6dce02039226bbd22aa43672b
```

Then:

```bash
git push   --force-with-lease=refs/heads/3D-test:f470cfd4e42d2ed6dce02039226bbd22aa43672b   origin HEAD:refs/heads/3D-test
```

If the lease fails, stop. Do not replace it with `--force`.

## After the rewrite

Immediately verify:

```bash
git fetch origin
git rev-parse origin/3D-test
git log --format=%B 5129422..origin/3D-test
```

Confirm:

- the remote SHA equals the rewritten `d7082f8...` SHA;
- there are no unwanted `Co-Authored-By`, `Claude`, or `Anthropic` strings in the active rewritten history.

Then report:

```text
remote SHA:
CI run:
CI status:
```

for the final C3.5 audit.
