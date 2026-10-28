"""A simulated user chats with the running app, then saves the conversation for error analysis.

1. Start the app in one terminal:                  python server.py
2. In another terminal, from the project root:     python -m evals.simulate_chat

Every run creates ONE new chat and sends 15 messages (change it with --turns), one at a time: the next message is
only written after the app has fully answered the last one. The messages come from an LLM that plays the same one
person every time (see evals/scenarios.py). Each chat has its own theme and drifts to other areas of life, and
themes and areas that earlier runs used are avoided, so run after run the chats stay different.
The results go to evals/runs/<time>/: transcript.md, review.csv and turns.json.
"""
import argparse
import csv
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import httpx
import litellm
from dotenv import load_dotenv

from evals.scenarios import AREAS, BEATS, OTHER_BEATS, PERSON, SYSTEM_PROMPT, THEMES

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(__file__).resolve().parent / "runs"
SIM_MODEL = "gemini/gemini-3.5-flash-lite"  # plays the user; not the model the app answers with
REPLY_CHARS_SEEN = 8000  # how much of each assistant reply the simulated user reads (long ones are marked as shortened)
CSV_COLUMNS = [
    "run", "chat_id", "turn", "intent", "area", "web_tool_ticked", "user_message", "steps", "assistant_reply",
    "seconds", "error", "trace_link", "what_went_wrong", "which_part",  # the last three are for your notes
]

load_dotenv(ROOT / ".env")
# SECRET CONSUMED HERE: GEMINI_API_KEY2, a key from a different Gemini account than the app's GEMINI_API_KEY, so the
# simulated user does not use up the app's quota (never printed)
SIM_API_KEY = os.environ.get("GEMINI_API_KEY2") or os.environ.get("GEMINI_API_KEY")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
for _noisy in ("httpx", "LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):  # keep the log to the steps of the run
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger("sim")


def _short(text: str, limit: int = 140) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def load_history() -> list[dict]:
    """The runs saved earlier, oldest first. They decide which theme and areas are used next."""
    runs = []
    for path in sorted(RUNS_DIR.glob("*/turns.json")):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return runs


def choose_theme(history: list[dict], name: str | None) -> dict:
    """The theme asked for, or one that no earlier run used (any theme once all have been used)."""
    if name:
        matches = [t for t in THEMES if t["name"] == name]
        if not matches:
            raise SystemExit(f"Unknown theme. Choose from: {', '.join(t['name'] for t in THEMES)}")
        return matches[0]
    used = [run.get("theme") for run in history]
    return random.choice([t for t in THEMES if t["name"] not in used] or THEMES)


def plan_beats(n: int) -> dict[int, str]:
    """Decide what each of the n messages does. Turns that get no beat just continue the conversation."""
    middle = list(range(1, n - 1))
    share = random.choice(middle[:4])  # a personal fact comes up early ...
    recall = random.choice(middle[-3:])  # ... and is needed again near the end
    free = [turn for turn in middle if turn not in (share, recall)]
    plan = {0: "opening", n - 1: "wrap_up", share: "memory_share", recall: "memory_recall"}
    plan.update(zip(random.sample(free, len(OTHER_BEATS)), OTHER_BEATS))
    return plan


def plan_turns(n: int, theme: dict, history: list[dict]) -> tuple[list[dict], list[str]]:
    """For each message: what it does (beat) and which area of life it is about.

    A chat starts on its theme, then moves to two other areas, preferring areas that earlier runs used least.
    When there are earlier chats, the late memory question may be about a fact told in one of them.
    """
    beats = plan_beats(n)
    used = [area for run in history for area in run.get("areas", [])]
    others = sorted((a for a in AREAS if a != theme["area"]), key=lambda a: (used.count(a), random.random()))[:2]
    areas = [theme["area"], *others]
    first_change = random.randint(n // 3 - 1, n // 3 + 1)
    second_change = random.randint(2 * n // 3 - 1, 2 * n // 3 + 1)
    earlier_facts = [
        t["user_message"] for run in history for t in run.get("turns", []) if t.get("intent") == "memory_share"
    ]
    plan = []
    for i in range(n):
        beat, fact = beats.get(i, "continue"), ""
        if beat == "memory_recall" and earlier_facts and random.random() < 0.5:
            beat, fact = "memory_recall_earlier", random.choice(earlier_facts)
        plan.append({"beat": beat, "fact": fact, "area": areas[0 if i < first_change else 1 if i < second_change else 2]})
    return plan, areas


def next_user_message(history: list[tuple[str, str]], step: dict, theme: dict, number: int, total: int) -> str:
    """Ask the simulated user for its next message, given everything said so far and what this message should do."""
    transcript = "\n".join(
        f"{'You' if role == 'user' else 'Assistant'}: "
        + (content if len(content) <= REPLY_CHARS_SEEN else content[:REPLY_CHARS_SEEN] + " [... shortened here only]")
        for role, content in history
    ) or "(nothing yet)"
    note = BEATS[step["beat"]].replace("{fact}", step["fact"])
    # WARNING: real external API call (uses free-tier quota)
    response = litellm.completion(
        model=SIM_MODEL,
        api_key=SIM_API_KEY,
        num_retries=2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.replace("{persona}", PERSON["bio"])},
            {
                "role": "user",
                "content": (
                    f"Conversation so far:\n{transcript}\n\nThis is message {number} of {total}.\n"
                    f"The main storyline of this chat: {theme['brief']}\n"
                    f"Subject area for this message: {step['area']}. If that is not what the last message was about, "
                    "change the subject the way a real person does.\n"
                    f"{note}\n\nWrite your next message."
                ),
            },
        ],
    )
    message = (response.choices[0].message.content or "").strip().strip('"').removeprefix("You:").strip()
    if not message:
        raise RuntimeError("The simulated user returned an empty message")
    return message


def send(client: httpx.Client, chat_id: str, content: str, use_web: bool) -> dict:
    """Send one message and wait for the app's whole answer: steps, reply, any error, and the end of the stream."""
    turn = {"steps": [], "reply": None, "error": None, "answered_at": None}  # answered_at: when the reply or error came
    started = time.perf_counter()
    try:
        # WARNING: this runs the full app: model calls, web searches, and writes to its database
        with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"content": content, "use_web": use_web}) as res:
            res.raise_for_status()
            for line in res.iter_lines():  # ends when the app has finished the whole turn, saving included
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "step":
                    turn["steps"].append(event["text"])
                elif event["type"] == "reply":
                    turn["reply"] = event["content"]
                    turn["answered_at"] = time.perf_counter()
                elif event["type"] == "error":
                    turn["error"] = event["message"]
                    turn["answered_at"] = time.perf_counter()
    except httpx.HTTPError as exc:
        turn["error"] = f"{type(exc).__name__}: {exc}"
    turn["seconds"] = round(time.perf_counter() - started, 1)
    return turn


def save_run(run_dir: Path, meta: dict, turns: list[dict]) -> None:
    """Write turns.json (everything), review.csv (a sheet to add notes to) and transcript.md (easy to read)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "turns.json").write_text(
        json.dumps({**meta, "turns": turns}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(run_dir / "review.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, CSV_COLUMNS)
        writer.writeheader()
        for t in turns:
            writer.writerow({
                "run": meta["run"], "chat_id": meta["chat_id"], "turn": t["turn"], "intent": t["intent"],
                "area": t["area"], "web_tool_ticked": "yes" if t["use_web"] else "no", "user_message": t["user_message"],
                "steps": " > ".join(t["steps"]), "assistant_reply": t["reply"] or "", "seconds": t["seconds"],
                "error": t["error"] or "", "trace_link": "", "what_went_wrong": "", "which_part": "",
            })
    lines = [
        f"# Run {meta['run']}", "",
        f"Chat `{meta['chat_id']}` · theme {meta['theme']} · areas {', '.join(meta['areas'])} · {len(turns)} messages", "",
    ]
    for t in turns:
        lines += [
            f"## Turn {t['turn']} ({t['intent']}, {t['area']})", "",
            f"**User** (web tool ticked: {'yes' if t['use_web'] else 'no'}): {t['user_message']}", "",
        ]
        if t["steps"]:
            lines += ["Steps: " + " > ".join(t["steps"]), ""]
        lines += [f"**Assistant** ({t['seconds']}s): {t['reply'] or '(no reply)'}", ""]
        if t["error"]:
            lines += [f"**ERROR:** {t['error']}", ""]
    (run_dir / "transcript.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the running app as a simulated user.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="where the app is running")
    parser.add_argument("--turns", type=int, default=15, help="messages to send in one run (at least 11)")
    parser.add_argument("--theme", help=f"one of {', '.join(t['name'] for t in THEMES)} (default: one not used before)")
    parser.add_argument("--web-prob", type=float, default=0.2, help="chance that Use web tool is ticked on a message")
    parser.add_argument("--pause", type=float, default=30.0, help="seconds between the assistant's reply and the next message")
    parser.add_argument("--seed", type=int, help="repeat the same plan and ticks")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("GEMINI_API_KEY2"):
        log.warning("GEMINI_API_KEY2 is not set in .env, so the simulated user shares the app's Gemini limit")
    if args.turns < 11:
        raise SystemExit("Use --turns 11 or more.")
    if args.seed is not None:
        random.seed(args.seed)
    history = load_history()
    theme = choose_theme(history, args.theme)
    total = args.turns
    plan, areas = plan_turns(total, theme, history)

    client = httpx.Client(base_url=args.url, timeout=httpx.Timeout(300, connect=10))
    try:
        client.get("/api/chats").raise_for_status()
    except httpx.HTTPError as exc:
        raise SystemExit(f"Cannot reach the app at {args.url} ({exc}). Start it first with: python server.py")
    if not args.yes:
        answer = input(
            f"This chats with the app at {args.url} as {PERSON['name']}: it creates one new chat and about {total * 2} messages in\n"
            "the app's database, and the assistant may save memories about this made-up person.\nContinue? [y/N] "
        )
        if answer.strip().lower() != "y":
            raise SystemExit("Cancelled.")

    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run
    # WARNING: writes one new row to public.chats (through the app)
    chat_id = client.post("/api/chats").raise_for_status().json()["chat_id"]
    meta = {"run": run, "chat_id": chat_id, "person": PERSON["name"], "theme": theme["name"], "areas": areas,
            "planned_turns": total, "started_at": datetime.now().isoformat(timespec="seconds"),
            "simulator_model": SIM_MODEL}
    log.info("Run %s | chat %s | %d earlier runs", run, chat_id, len(history))
    log.info("Theme: %s | areas: %s", theme["name"], ", ".join(areas))
    log.info("%d messages, web tool ticked on about %d%% of them", total, round(args.web_prob * 100))

    messages: list[tuple[str, str]] = []
    turns: list[dict] = []
    shared_text = ""  # what the person wrote when sharing a fact, so the later recall question is about something real
    try:
        for index, step in enumerate(plan):
            if step["beat"] == "memory_recall":
                step["fact"] = shared_text
            use_web = random.random() < args.web_prob
            message = next_user_message(messages, step, theme, index + 1, total)
            log.info("Message %d of %d (%s, %s) | web tool ticked: %s",
                     index + 1, total, step["beat"], step["area"], "yes" if use_web else "no")
            log.info("    User: %s", _short(message))
            if step["beat"] == "memory_share":
                shared_text = message
            result = send(client, chat_id, message, use_web)  # waits for the whole answer
            answered_at = result.pop("answered_at")
            for text in result["steps"]:
                log.info("      step: %s", text)
            log.info("    Assistant (%.1fs): %s", result["seconds"], _short(result["reply"] or "(no reply)"))
            if result["error"]:
                log.info("    ERROR: %s", _short(result["error"], 300))
            turns.append({"turn": index + 1, "intent": step["beat"], "area": step["area"], "use_web": use_web,
                          "user_message": message, **result})
            messages += [("user", message), ("assistant", result["reply"] or "[the assistant did not reply: an error happened]")]
            if index < total - 1:
                # The gap counts from the reply, not from the end of the turn: only the reply used Groq, and the
                # saving and the summary that follow do not. The next message still waits for the turn to finish.
                wait = max(0.0, args.pause - (time.perf_counter() - answered_at)) if answered_at else args.pause
                log.info("    Waiting %d more s, so the next message comes %d s after the reply", wait, args.pause)
                time.sleep(wait)
    except KeyboardInterrupt:
        log.info("Stopped early, saving what was said so far")
    finally:
        meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        save_run(run_dir, meta, turns)
        failed = sum(1 for t in turns if t["error"])
        log.info("Saved %d messages (%d with errors) to %s", len(turns), failed, run_dir)


if __name__ == "__main__":
    main()
