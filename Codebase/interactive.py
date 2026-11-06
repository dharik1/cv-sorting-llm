"""
interactive.py - Recruiter-in-the-loop refinement of the ranking (terminal command loop).

LLM-2 stores one score per requirement, so changing weights or priorities and adding hard skill
filters re-rank instantly without any model call. Only `add` (a brand-new requirement) calls
LLM-2 again, and only for that single requirement (results are cached).
"""
from __future__ import annotations

import os

from ranking import export_ranking_csv, format_candidate, format_ranking, format_requirements, rank_candidates

HELP = """Commands:
  list [N]                show the current ranking (top N, default 10)
  show <rank|file>        requirement-by-requirement evidence for one candidate
  reqs                    list requirements with priority and weight
  weight <ID> <w>         change a requirement's weight (0 = ignore it)
  must <ID> | nice <ID>   change a requirement's priority
  require <skill>         keep only candidates having this skill (synonym aware)
  clear                   remove all skill filters
  add must|nice <text>    add a new requirement (LLM-2 scores it for every shortlisted CV)
  save                    export the refined ranking to ranking_refined.csv
  help | quit"""


class InteractiveSession:
    """Terminal loop through which a recruiter inspects and refines the ranking."""

    def __init__(self, candidates, job, config, judge, out_dir, must_weight, nice_weight):
        """Keep the scored candidates, job requirements, scoring settings and the LLM-2 judge."""
        self.candidates = candidates
        self.job = job
        self.config = config
        self.judge = judge
        self.out_dir = out_dir
        self.weights = {"must_have": must_weight, "nice_to_have": nice_weight}
        self.required_skills = []
        self.ranked = rank_candidates(candidates, job["requirements"], config)

    @property
    def requirements(self) -> list:
        """The current (possibly edited) requirement checklist."""
        return self.job["requirements"]

    def run(self) -> None:
        """Read and execute commands until 'quit' (Ctrl-D / Ctrl-C also exit)."""
        print("\nInteractive refinement mode - type 'help' for the list of commands.")
        while True:
            try:
                line = input("refine> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            command, _, argument = line.partition(" ")
            command = command.lower()
            if not command:
                continue
            if command in ("quit", "exit", "q"):
                break
            handler = getattr(self, f"_cmd_{command}", None)
            if handler is None:
                print("Unknown command - type 'help'.")
                continue
            try:
                handler(argument.strip())
            except (ValueError, IndexError, KeyError) as exc:
                print(f"Invalid input: {exc}")

    def _rerank(self) -> None:
        """Recompute the ranking with the current settings and print the top 10."""
        self.ranked = rank_candidates(self.candidates, self.requirements, self.config, self.required_skills)
        print(format_ranking(self.ranked, self.requirements, 10))

    def _requirement(self, req_id: str) -> dict:
        """Find a requirement by id (case-insensitive) or raise KeyError."""
        for r in self.requirements:
            if r["id"].lower() == req_id.lower():
                return r
        raise KeyError(f"no requirement '{req_id}' (type 'reqs' to list them)")

    def _cmd_help(self, _argument: str) -> None:
        """Print the command reference."""
        print(HELP)

    def _cmd_list(self, argument: str) -> None:
        """Print the top-N ranking."""
        print(format_ranking(self.ranked, self.requirements, int(argument) if argument else 10))

    def _cmd_reqs(self, _argument: str) -> None:
        """Print the requirement checklist."""
        print(format_requirements(self.requirements))

    def _cmd_show(self, argument: str) -> None:
        """Print the evidence for the candidate at a visible rank, or whose file name contains the text."""
        visible = [c for c in self.ranked if not c.filtered_out]
        if argument.isdigit():
            candidate = visible[int(argument) - 1]
        else:
            matches = [c for c in self.candidates if argument.lower() in c.file.lower()]
            if not matches:
                raise KeyError(f"no CV file matching '{argument}'")
            candidate = matches[0]
        print(format_candidate(candidate, self.requirements))

    def _cmd_weight(self, argument: str) -> None:
        """Change one requirement's weight and re-rank."""
        req_id, value = argument.split()
        self._requirement(req_id)["weight"] = max(0.0, float(value))
        self._rerank()

    def _set_priority(self, req_id: str, priority: str) -> None:
        """Change one requirement's priority (resetting its weight to that priority's default) and re-rank."""
        requirement = self._requirement(req_id)
        requirement["priority"] = priority
        requirement["weight"] = self.weights[priority]
        self._rerank()

    def _cmd_must(self, argument: str) -> None:
        """Mark a requirement as must-have."""
        self._set_priority(argument, "must_have")

    def _cmd_nice(self, argument: str) -> None:
        """Mark a requirement as nice-to-have."""
        self._set_priority(argument, "nice_to_have")

    def _cmd_require(self, argument: str) -> None:
        """Add a hard skill filter and re-rank."""
        if not argument:
            raise ValueError("usage: require <skill>")
        self.required_skills.append(argument)
        self._rerank()

    def _cmd_clear(self, _argument: str) -> None:
        """Remove all skill filters and re-rank."""
        self.required_skills = []
        self._rerank()

    def _cmd_add(self, argument: str) -> None:
        """Add a new requirement, let LLM-2 score it for every shortlisted candidate, then re-rank."""
        level, _, text = argument.partition(" ")
        priority = {"must": "must_have", "nice": "nice_to_have"}.get(level.lower())
        if priority is None or not text.strip():
            raise ValueError("usage: add must|nice <requirement text>")
        requirement = {"id": f"R{len(self.requirements) + 1}", "text": text.strip(), "priority": priority,
                       "category": "other", "weight": self.weights[priority]}
        shortlisted = [c for c in self.candidates if c.shortlisted]
        for i, candidate in enumerate(shortlisted, 1):
            print(f"  LLM-2 scoring {requirement['id']} for {candidate.file} ({i}/{len(shortlisted)})...", flush=True)
            result = self.judge.assess(candidate, [requirement], self.job["job_title"], with_summary=False)
            candidate.assessments.update(result["assessments"])
        self.requirements.append(requirement)
        self._rerank()

    def _cmd_save(self, _argument: str) -> None:
        """Export the current ranking to CSV."""
        path = os.path.join(self.out_dir, "ranking_refined.csv")
        export_ranking_csv(path, self.ranked, self.requirements)
        print(f"Saved {path}")
