#!/usr/bin/env python3
"""Check: does every code commit say who wrote it?

AGENTS.md rule 3: the lead does not write code -- every code ticket goes to DeepSeek
through DSH, and a fallback (Sonnet, Haiku, the lead) is named with its reason and its
evidence. This is that rule as a check, because a reminder alone does not hold it
(.claude/hooks/prior_work_gate.py, its docstring):

    python scripts/check_worker_trailer.py --message-file <git message file>
    python scripts/check_worker_trailer.py --range <before>..<after> [--repo <dir>]

--message-file is the hook mode, run by scripts/githooks/commit-msg: the commit is judged
by the files staged in the repo in the current directory. A merge commit, and a commit
with no code file staged, pass without a trailer; nothing is printed when a commit passes.

--range is the CI mode, run by .github/worker-trailer/check_worker_trailer.py on GitHub:
every commit in "git rev-list <before>..<after>" that changes code is judged, merge
commits are skipped, and a commit whose author date is before CUTOFF is exempt, so old
history pushed again is not failed. One line per failing commit: short sha, subject,
reason.

What counts as code, what the trailer must say, and the valid forms are in the ticket
.scratch/delegation-guardrails/issues/02-worker-trailer-gate.md, "Decisions" A.

Read-only: this script writes nothing, commits nothing and changes no git state.
Exit 0 = pass; 1 = refuse; 2 = the check could not run (bad arguments, git failed).
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

# Any of these extensions (case-insensitive) makes a changed path a code change.
CODE_EXTENSIONS = (
    ".py", ".sh", ".bash", ".slurm", ".sbatch", ".ps1", ".bat", ".cmd",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".cu",
    ".js", ".mjs", ".ts", ".tsx",
    ".m", ".r", ".ipynb",
)

# The shared hooks live here, so a change to them is a change to every clone's git.
GITHOOKS_DIR = ("scripts", "githooks")

# A path with any of these components is exempt, extension or not.
EXEMPT_COMPONENTS = (".scratch", "intent", ".progress")

# Commits authored before this instant are exempt: old history pushed again is not failed.
CUTOFF = datetime(2026, 11, 18, tzinfo=timezone.utc)

MAX_LISTED = 10
POINTER = '.claude/rules/claude_staffing.md, "Who does what"'
FALLBACK_REASONS = ("laptop off", "dsh down", "flash failed twice")
SMALL_OR_URGENT = re.compile(r"user called it small/urgent \d{4}-\d{2}-\d{2}")
_USER_AGREED = re.compile(r"user agreed \d{4}-\d{2}-\d{2}", re.IGNORECASE)
_LAPSE = re.compile(r"lapse\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
_SMALL_CHANGE = re.compile(r"small change\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
GIT_TIMEOUT = 120

OK, REFUSE, UNCHECKED = 0, 1, 2

VALID_FORMS = (
    "    Worker: DeepSeek",
    "    Worker: DeepSeek -- <anything>",
    "    Worker: Sonnet -- <reason>: <evidence>",
    "    Worker: Haiku -- <reason>: <evidence>",
    "    Worker: lead -- user agreed YYYY-MM-DD",
    "    Worker: lead -- lapse: <why>",
    "    Worker: lead -- small change: <diff size, review finding, or the user's words>",
    "    Worker: person",
    "    Worker: person -- <anything>",
)


def git(repo, *args, stdin=None):
    """(returncode, stdout bytes, stderr text); a git failure is a value, never a raise."""
    try:
        proc = subprocess.run(["git", "-C", repo] + [str(a) for a in args],
                              input=stdin, capture_output=True, timeout=GIT_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, b"", "%s: %s" % (exc.__class__.__name__, exc)
    return proc.returncode, proc.stdout, proc.stderr.decode("utf-8", "replace").strip()


def nul_names(data):
    """Split "git ... -z" output into names: NUL-separated, UTF-8, spaces kept."""
    return [piece.decode("utf-8", "replace") for piece in data.split(b"\0") if piece]


def components(path):
    return [part for part in path.replace("\\", "/").split("/") if part]


def is_code(path):
    """True when a changed path is a code change this rule gates."""
    parts = components(path)
    if any(part in EXEMPT_COMPONENTS for part in parts):
        return False
    for index, part in enumerate(parts[:-1]):
        if (part, parts[index + 1]) == GITHOOKS_DIR:
            return True
    return os.path.splitext(parts[-1])[1].lower() in CODE_EXTENSIONS


def read_message(path):
    with open(path, "rb") as handle:
        return handle.read().decode("utf-8", "replace")


def subject_of(message):
    """The first line of the message that is not a comment: the commit's subject."""
    for line in message.splitlines():
        if line.startswith("#"):
            continue
        if line.strip():
            return line.strip()
    return "(no subject)"


def parse_trailers(repo, message):
    """[(key, value)] of the message's final trailer block, or None if git could not read it.

    Lines starting with "#" are dropped first: the commit-msg hook runs before git strips
    them, so the comment column would otherwise sit in the last paragraph.
    """
    kept = [line for line in message.splitlines() if not line.startswith("#")]
    payload = ("\n".join(kept) + "\n").encode("utf-8")
    code, out, err = git(repo, "interpret-trailers", "--parse", stdin=payload)
    if code:
        return None
    parsed = []
    for line in out.decode("utf-8", "replace").splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip():
            parsed.append((key.strip(), value.strip()))
    return parsed


def normalise(value):
    """En and em dashes stand in for "--"; surrounding spaces are free."""
    return value.replace("\u2013", "--").replace("\u2014", "--").strip()


def split_dashes(text):
    """(who, rest or None): the part before the first "--", and the part after it."""
    index = text.find("--")
    if index < 0:
        return text.strip(), None
    return text[:index].strip(), text[index + 2:].strip()


def fallback_reason(reason):
    """The fallback's reason must be exactly one of the four in the table."""
    text = " ".join(reason.split()).casefold()
    return text in FALLBACK_REASONS or bool(SMALL_OR_URGENT.fullmatch(text))


def trailer_verdict(value):
    """None when the value is not a valid form; "pass", or "lapse" for a recorded lapse."""
    who, rest = split_dashes(normalise(value))
    who = who.casefold()

    if who in ("deepseek", "person"):
        return "pass"

    if who in ("sonnet", "haiku"):
        if rest is None:
            return None
        reason, sep, evidence = rest.partition(":")
        if not sep or not evidence.strip():
            return None
        return "pass" if fallback_reason(reason) else None

    if who == "lead":
        if rest is None:
            return None
        small = _SMALL_CHANGE.fullmatch(rest)
        if small and small.group(1).strip():
            return "pass"
        if _USER_AGREED.fullmatch(rest):
            return "pass"
        lapse = _LAPSE.fullmatch(rest)
        if lapse and lapse.group(1).strip():
            return "lapse"
        return None

    return None


def judge(repo, message):
    """(ok, reason, lapse_value) for one commit message.

    Every Worker trailer in the final block must be valid when there is more than one.
    """
    parsed = parse_trailers(repo, message)
    if parsed is None:
        return False, "git interpret-trailers could not read its message", None
    workers = [value for key, value in parsed if key.casefold() == "worker"]
    if not workers:
        return False, "no Worker: trailer in its final trailer block", None
    lapse = None
    for value in workers:
        verdict = trailer_verdict(value)
        if verdict is None:
            return False, 'its Worker: trailer "%s" is not one of the valid forms' % value, None
        if verdict == "lapse":
            lapse = value
    return True, None, lapse


def warn_lapse(subject, value):
    """A lapse passes, but it must be visible in the terminal that made the commit."""
    sys.stderr.write("warning: recorded lapse\n  subject: %s\n  trailer: Worker: %s\n"
                     % (subject, value))


def refusal(files, reason):
    """The text a refused commit sees: the files, the reason, and the forms that pass."""
    lines = [
        "Commit refused: this commit changes code, but its message does not say who wrote it.",
        "",
        "  " + reason,
        "",
        "Code files in this commit:",
    ]
    for path in files[:MAX_LISTED]:
        lines.append("    " + path)
    if len(files) > MAX_LISTED:
        lines.append("    ... and %d more" % (len(files) - MAX_LISTED))
    lines += [
        "",
        "End the message with a Worker: trailer that names its author. These pass:",
        "",
    ]
    lines += list(VALID_FORMS)
    lines += [
        "",
        "<reason> is exactly one of: laptop off, DSH down, Flash failed twice,",
        "user called it small/urgent YYYY-MM-DD. <evidence> says what happened and must",
        "not be empty. -- may be written as an en dash or an em dash.",
        "",
        "Who writes code, and the evidence a fallback needs:",
        POINTER,
    ]
    return "\n".join(lines) + "\n"


def staged_code(repo):
    """([code paths staged], error text or None)."""
    code, out, err = git(repo, "diff", "--cached", "--name-only", "-z")
    if code:
        return None, err or ("git diff --cached exited %d" % code)
    return [name for name in nul_names(out) if is_code(name)], None


def git_dir_of(repo):
    for option in ("--absolute-git-dir", "--git-dir"):
        code, out, _ = git(repo, "rev-parse", option)
        if code == 0 and out.strip():
            return out.decode("utf-8", "replace").strip()
    return ".git"


def hook_mode(repo, message_path):
    """The commit-msg mode: judge the commit that is being made in "repo"."""
    if os.path.exists(os.path.join(git_dir_of(repo), "MERGE_HEAD")):
        return OK                       # a merge commit carries no new code of its own

    files, error = staged_code(repo)
    if files is None:
        sys.stderr.write("Commit refused: the staged files could not be listed (%s).\n\n%s\n"
                         % (error, POINTER))
        return REFUSE
    if not files:
        return OK                       # no code in this commit: nothing to attribute

    try:
        message = read_message(message_path)
    except OSError as exc:
        sys.stderr.write("Commit refused: the commit message could not be read (%s).\n\n%s\n"
                         % (exc, POINTER))
        return REFUSE

    ok, reason, lapse = judge(repo, message)
    if not ok:
        sys.stderr.write(refusal(files, reason))
        return REFUSE
    if lapse:
        warn_lapse(subject_of(message), lapse)
    return OK


def all_zero(sha):
    return bool(sha) and set(sha) == {"0"}


def parse_iso(text):
    """A git ISO 8601 date as an aware UTC datetime, or None when it cannot be read."""
    text = (text or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def author_date(repo, rev):
    """The commit's author date in UTC -- never its committer date, which a rebase resets."""
    code, out, _ = git(repo, "log", "-1", "--format=%aI", rev)
    if code:
        return None
    return parse_iso(out.decode("ascii", "replace"))


def changed_code(repo, rev):
    """[code paths a commit changed] against its first parent; [] when it changed none."""
    code, out, err = git(repo, "diff-tree", "--root", "--no-commit-id", "-r",
                         "--name-only", "-z", rev)
    if code:
        return None
    return [name for name in nul_names(out) if is_code(name)]


def commit_message(repo, rev):
    code, out, err = git(repo, "log", "-1", "--format=%B", rev)
    return out.decode("utf-8", "replace") if code == 0 else None


def range_mode(repo, spec):
    """The CI mode: judge every non-merge commit in "<before>..<after>" after CUTOFF."""
    before, sep, after = spec.partition("..")
    before, after = before.strip(), after.strip()
    if not sep:
        sys.stderr.write("bad --range %r: expected <before>..<after>\n" % spec)
        return UNCHECKED
    if not after or all_zero(after):
        print("worker-trailer: nothing to check in %s" % spec)
        return OK                       # a branch deletion carries no commit

    if all_zero(before):
        # A new branch: only the commit <after> points at. Resolve it, so a symbolic name
        # (HEAD, a branch) still prints a short sha in the failure line.
        code, out, err = git(repo, "rev-parse", "--verify", after + "^{commit}")
        if code:
            sys.stderr.write("git rev-parse %s failed: %s\n" % (after, err))
            return UNCHECKED
        revs = [out.decode("ascii", "replace").strip()]
    else:
        code, out, err = git(repo, "rev-list", "%s..%s" % (before, after))
        if code:
            sys.stderr.write("git rev-list %s..%s failed: %s\n" % (before, after, err))
            return UNCHECKED
        revs = out.decode("ascii", "replace").split()

    checked, failures = 0, []
    for rev in revs:
        code, out, err = git(repo, "rev-list", "--parents", "-n", "1", rev)
        if code:
            sys.stderr.write("git rev-list --parents %s failed: %s\n" % (rev, err))
            return UNCHECKED
        if len(out.split()) > 2:
            continue                    # a merge commit carries no new code of its own

        stamped = author_date(repo, rev)
        if stamped is not None and stamped < CUTOFF:
            continue                    # authored before the rule existed: not failed

        files = changed_code(repo, rev)
        if files is None:
            sys.stderr.write("git diff-tree %s failed\n" % rev)
            return UNCHECKED
        if not files:
            continue

        message = commit_message(repo, rev)
        if message is None:
            sys.stderr.write("git log %s failed\n" % rev)
            return UNCHECKED

        checked += 1
        ok, reason, lapse = judge(repo, message)
        if not ok:
            failures.append((rev[:7], subject_of(message), reason))
        elif lapse:
            warn_lapse(subject_of(message), lapse)

    for short, subject, reason in failures:
        print("%s  %s  -- %s" % (short, subject, reason))
    if failures:
        sys.stderr.write("worker-trailer: %d of %d code commit(s) in %s carry no valid "
                         "Worker: trailer; %s\n" % (len(failures), checked, spec, POINTER))
        return REFUSE
    print("worker-trailer: %d code commit(s) in %s, all carry a valid Worker: trailer"
          % (checked, spec))
    return OK


def main(argv=None):
    # File names may be non-ASCII, and a commit must never be refused because a name could
    # not be printed on the console's code page.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(
        description="Check that a code commit says who wrote it (a Worker: trailer).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--message-file", metavar="PATH",
                       help="hook mode: judge the commit staged in the current repo")
    group.add_argument("--range", dest="range_spec", metavar="BEFORE..AFTER",
                       help="CI mode: judge every commit in this range")
    parser.add_argument("--repo", default=".",
                        help="the repository --range runs in (default: the current directory)")
    args = parser.parse_args(argv)

    if args.message_file is not None:
        return hook_mode(".", args.message_file)
    return range_mode(args.repo, args.range_spec)


if __name__ == "__main__":
    sys.exit(main())
