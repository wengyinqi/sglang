"""Synchronize only five official refs; experiment refs are never updated.

Default: fetch and show the plan. --apply updates unoccupied local mirror refs.
--push (requires --apply) updates the personal fork with explicit leases. If a
mirror contains divergent work, preserve it under another branch before sync.
"""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRANCHES = (
    "main",
    "hisparse-fix",
    "hisparse_quest",
    "hisparse-prefetch-indexshare",
    "hisparse-prefetch-indexshare-io-only",
)
UPSTREAM = "https://github.com/sgl-project/sglang.git"


def git(*args, check=True):
    return subprocess.run(
        ["git", "-c", "http.version=HTTP/1.1", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--push", action="store_true")
    args = p.parse_args(argv)
    if args.push and not args.apply:
        p.error("--push requires --apply")
    if git("remote", "get-url", "upstream").stdout.strip().removesuffix(
        ".git"
    ) != UPSTREAM.removesuffix(".git"):
        raise RuntimeError(
            "upstream must point to the official sgl-project/sglang repository"
        )
    git(
        "fetch",
        "--no-tags",
        "upstream",
        *(f"+refs/heads/{name}:refs/remotes/upstream/{name}" for name in BRANCHES),
    )
    occupied = {
        line.removeprefix("branch refs/heads/")
        for line in git("worktree", "list", "--porcelain").stdout.splitlines()
        if line.startswith("branch refs/heads/")
    }
    plan = []
    for name in BRANCHES:
        old = git("rev-parse", f"refs/heads/{name}").stdout.strip()
        new = git("rev-parse", f"refs/remotes/upstream/{name}").stdout.strip()
        fast_forward = (
            git("merge-base", "--is-ancestor", old, new, check=False).returncode == 0
        )
        # A rewind is safe only when every old commit has an explicit durable
        # non-mirror branch, never merely a reflog or the mirror itself.
        preserved = [
            ref
            for ref in git(
                "for-each-ref",
                f"--contains={old}",
                "--format=%(refname:short)",
                "refs/heads",
            ).stdout.splitlines()
            if ref not in BRANCHES
        ]
        plan.append(
            dict(
                branch=name,
                old=old,
                new=new,
                fast_forward=fast_forward,
                preserved_by=preserved,
            )
        )
    print(json.dumps(plan, indent=2))
    if not args.apply:
        return
    if occupied.intersection(BRANCHES):
        raise RuntimeError(
            f"Switch worktrees off official mirrors before applying: {sorted(occupied.intersection(BRANCHES))}"
        )
    if any(not item["fast_forward"] and not item["preserved_by"] for item in plan):
        raise RuntimeError(
            "Divergent mirror work must be preserved under another branch before applying"
        )
    remote = {}
    if args.push:
        for line in git(
            "ls-remote", "--heads", "origin", *BRANCHES
        ).stdout.splitlines():
            sha, ref = line.split()
            remote[ref.removeprefix("refs/heads/")] = sha
        # A remote-only update must be inspected/fetched rather than silently
        # overwritten by a successful local mirror update.
        if any(
            remote.get(item["branch"]) not in (item["old"], item["new"])
            for item in plan
        ):
            raise RuntimeError(
                "Origin contains uninspected mirror changes; fetch and preserve them first"
            )
    commands = (
        "start\n"
        + "".join(
            f"update refs/heads/{item['branch']} {item['new']} {item['old']}\n"
            for item in plan
        )
        + "prepare\ncommit\n"
    )
    subprocess.run(
        ["git", "update-ref", "--stdin"],
        cwd=ROOT,
        # Git's ref protocol requires LF even on Windows; text-mode pipes
        # translate it to CRLF, making commands such as "start" invalid.
        input=commands.encode("ascii"),
        check=True,
    )
    for name in BRANCHES:
        git("branch", f"--set-upstream-to=upstream/{name}", name)
    if args.push:
        leases = [
            f"--force-with-lease=refs/heads/{name}:{remote[name]}" for name in BRANCHES
        ]
        git(
            "push",
            "--atomic",
            *leases,
            "origin",
            *(f"refs/heads/{name}:refs/heads/{name}" for name in BRANCHES),
        )


if __name__ == "__main__":
    main()
