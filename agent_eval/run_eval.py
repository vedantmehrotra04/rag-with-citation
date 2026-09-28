import json
import time
from pathlib import Path

from agent.runner import run_agent
from agent_eval.scorers import score_all

DELAY = 7.0          # Cohere trial key allows 10 calls/minute


def main() -> None:
    # Reset the credit log — a credit left over from a previous run makes
    # already_credited True and the agent behaves differently. Same scenario,
    # different answer, and nothing on screen says why.
    Path("credits.json").unlink(missing_ok=True)

    scenarios = json.load(open("agent_eval/scenarios.json"))
    results = []

    for i, scenario in enumerate(scenarios, start=1):
        print(f"[{i:2}/{len(scenarios)}] {scenario['id']}")
        try:
            trace = run_agent(scenario["input"])
        except Exception as exc:                     # a crash is a result, not a stop
            trace = {
                "input": scenario["input"],
                "steps": [],
                "final_answer": f"ERROR: {exc}",
                "terminated": "error",
                "outcome": "error",
            }
            print(f"          crashed: {exc}")

        results.append((scenario, trace))
        time.sleep(DELAY)

    print()
    summary = score_all(results)

    Path("agent_eval/last_run.json").write_text(
        json.dumps([{"scenario": s, "trace": t} for s, t in results], indent=2, default=str)
    )
    print("\ntraces written to agent_eval/last_run.json")
    print(summary)


if __name__ == "__main__":
    main()