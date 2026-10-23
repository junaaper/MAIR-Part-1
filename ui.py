# Window for the usability study: shows the participant's scenario, the conversation in large text,
# reads the system utterances aloud (optional) and logs every turn, skipped speech and the conversation time
# Usage: python ui.py --scenario scenarios/scenario_a.json [--participant P01] [--tts]
#                     [--transparency on|off] [--recommendation-mode patient|eager] [other dialog_manager options]
import os
import sys
import json
import time
import queue
import argparse
import threading
import tkinter as tk

import pyglet

import dialog_manager as dm
from dialog_acts import clean_utterance
from classifiers import load_model, predict
from slot_extraction import MATCHERS, warm_up

FONT = "Arial"
TEXT_SIZE = 18
BACKGROUND = "#ffffff"
SCENARIO_BACKGROUND = "#fff6d6"
SYSTEM_COLOR = "#0b3d91"
USER_COLOR = "#1b5e20"
POLL_MS = 50  # how often the window checks for finished speech synthesis and updates the audio


def load_scenario(path):
    """Read a scenario file, stops with a readable message if it can't be used."""
    try:
        with open(path, encoding="utf-8") as f:
            scenario = json.load(f)
    except (OSError, json.JSONDecodeError) as error:
        sys.exit(f"Error: could not read scenario file '{path}' ({error})")
    missing = [key for key in ("id", "title", "story", "wishes") if key not in scenario]
    if missing:
        sys.exit(f"Error: scenario file '{path}' is missing: {', '.join(missing)}")
    return scenario


class DialogWindow:
    """The study window: scenario on top, conversation in the middle, input at the bottom."""

    def __init__(self, root, args, scenario, model):
        self.program_start = time.monotonic()  # the conversation time runs from here until goodbye
        self.root, self.args, self.scenario, self.model = root, args, scenario, model
        self.turn = 0
        self.ended = False
        self.player = None          # pyglet player of the utterance that is being read aloud
        self.speech_id = 0          # number of the latest utterance sent to TTS, older ones are ignored
        self.speech_pending = False  # True while the latest utterance is still being synthesized
        self.speech_queue = queue.Queue()

        self.log_path = None if args.no_log else dm.start_log(
            {"interface": "window", "participant": args.participant, "scenario": scenario["id"],
             "tts": args.tts, "classifier": dm.CLASSIFIER, "matcher": args.matcher,
             "transparency": args.transparency == "on", "seed": args.seed,
             "reasoning": dm.REASONING_AVAILABLE, "recommendation_mode": args.recommendation_mode,
             "restaurants": args.restaurants or "default"}, args.log_dir)

        self._build()
        self.state, reply = dm.start(args.matcher, args.seed, args.transparency == "on",
                                     args.recommendation_mode)
        self._system_says(reply)
        dm.log(self.log_path, {"event": "system", "turn": 0, "state": self.state.state, "utterance": reply})
        self.root.after(POLL_MS, self._poll)

    # Building the window

    def _build(self):
        root = self.root
        root.title("Cambridge restaurant system")
        root.configure(bg=BACKGROUND)
        root.geometry("1100x900")
        root.minsize(800, 650)
        try:
            root.state("zoomed")  # start maximized (Windows), so everything is as large as possible
        except tk.TclError:
            pass
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        tk.Label(root, text="Welcome to the Cambridge restaurant system", font=(FONT, 26, "bold"),
                 bg=BACKGROUND).pack(pady=(16, 8))

        box = tk.Frame(root, bg=SCENARIO_BACKGROUND, highlightbackground="#c9a227", highlightthickness=2)
        box.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(box, text=f"Your scenario: {self.scenario['title']}", font=(FONT, 20, "bold"),
                 bg=SCENARIO_BACKGROUND, anchor="w").pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(box, text=self.scenario["story"], font=(FONT, TEXT_SIZE), bg=SCENARIO_BACKGROUND,
                 justify="left", anchor="w", wraplength=1000).pack(fill="x", padx=14)
        wishes = "\n".join(f"•  {wish}" for wish in self.scenario["wishes"])
        tk.Label(box, text="What you want:\n" + wishes, font=(FONT, TEXT_SIZE), bg=SCENARIO_BACKGROUND,
                 justify="left", anchor="w").pack(fill="x", padx=14, pady=(6, 10))

        # The hint and the input row are packed at the bottom first, so they always stay visible
        hint = "Type your answer and press Enter (or Send)."
        if self.args.tts:
            hint += "  To stop the voice and just read, press Enter or Esc."
        self.hint = tk.Label(root, text=hint, font=(FONT, 16), bg=BACKGROUND, fg="#444444")
        self.hint.pack(side="bottom", pady=(0, 14))
        bottom = tk.Frame(root, bg=BACKGROUND)
        bottom.pack(side="bottom", fill="x", padx=20, pady=(10, 4))
        self.entry = tk.Entry(bottom, font=(FONT, 22), relief="solid", borderwidth=2)
        self.entry.pack(side="left", fill="x", expand=True, ipady=8)
        self.send_button = tk.Button(bottom, text="Send", font=(FONT, 20, "bold"), width=8,
                                     command=self._on_enter)
        self.send_button.pack(side="left", padx=(10, 0))
        self.stop_button = tk.Button(bottom, text="Stop the voice", font=(FONT, 20), width=14,
                                     command=lambda: self._stop_speaking("button"))
        if self.args.tts:
            self.stop_button.pack(side="left", padx=(10, 0))

        chat_frame = tk.Frame(root, bg=BACKGROUND)
        chat_frame.pack(fill="both", expand=True, padx=20)
        scrollbar = tk.Scrollbar(chat_frame, width=22)
        scrollbar.pack(side="right", fill="y")
        self.chat = tk.Text(chat_frame, font=(FONT, TEXT_SIZE), wrap="word", bg=BACKGROUND, relief="solid",
                            borderwidth=1, padx=12, pady=10, spacing3=10, height=6,
                            yscrollcommand=scrollbar.set)
        self.chat.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.chat.yview)
        self.chat.tag_configure("system_name", foreground=SYSTEM_COLOR, font=(FONT, TEXT_SIZE, "bold"))
        self.chat.tag_configure("user_name", foreground=USER_COLOR, font=(FONT, TEXT_SIZE, "bold"))
        self.chat.configure(state="disabled")

        root.bind("<Return>", lambda event: self._on_enter())
        root.bind("<KP_Enter>", lambda event: self._on_enter())
        root.bind("<Escape>", lambda event: self._stop_speaking("escape"))
        self.entry.focus_set()

    # Conversation

    def _add_message(self, speaker, text):
        self.chat.configure(state="normal")
        self.chat.insert("end", f"{speaker}: ", "system_name" if speaker == "System" else "user_name")
        self.chat.insert("end", text + "\n")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _system_says(self, reply):
        self._add_message("System", reply)
        self._speak(reply)

    def _on_enter(self):
        """Enter or Send: an empty Enter only stops the voice, otherwise the answer is processed."""
        if self.ended:
            return
        text = self.entry.get().strip()
        self._stop_speaking("enter" if text == "" else "answer")
        if text == "":
            return
        self.entry.delete(0, "end")
        self._add_message("You", text)

        utterance = clean_utterance(text)
        if utterance == "":
            self._system_says("Please type a sentence.")
            return
        self.turn += 1
        act = predict(dm.CLASSIFIER, self.model, [utterance])[0]
        previous = self.state.state
        self.state, reply = dm.transition(self.state, act, text)
        self._system_says(reply)
        dm.log(self.log_path, {"event": "turn", "turn": self.turn, "user": text, "act": act,
                               "state_before": previous, "state_after": self.state.state,
                               "prefs": self.state.prefs, "extra": self.state.extra,
                               "suggested": self.state.current["restaurantname"] if self.state.current else None,
                               "system": reply})
        if self.state.state == dm.FINISHED:
            self._end("goodbye")

    def _end(self, reason):
        """Write the end of the conversation to the log (once) and close the input."""
        if self.ended:
            return
        self.ended = True
        suggested = self.state.current["restaurantname"] if self.state.current else None
        correct = self.scenario.get("correct_restaurants")
        dm.log(self.log_path, {"event": "end", "reason": reason, "turns": self.turn,
                               "duration_seconds": round(time.monotonic() - self.program_start, 1),
                               "scenario": self.scenario["id"], "state": self.state.state,
                               "prefs": self.state.prefs, "extra": self.state.extra, "suggested": suggested,
                               "task_success": suggested in correct if correct and suggested else False})
        self.entry.configure(state="disabled")
        self.send_button.configure(state="disabled")
        self.hint.configure(text="The conversation has ended. Thank you! You can close this window now.",
                            fg="#000000", font=(FONT, 18, "bold"))

    def _on_close(self):
        self._stop_speaking("window closed")
        self._end("window closed")
        self.root.destroy()

    # Speech

    def _speak(self, text):
        """Read the text aloud in the background, so the window never freezes."""
        if not self.args.tts:
            return
        self._stop_speaking(None)  # a new utterance replaces the previous one (not a skip)
        self.speech_id += 1
        self.speech_pending = True
        speech_id = self.speech_id

        def synthesize():
            try:
                self.speech_queue.put((speech_id, dm.tts_mp3(text), None))
            except Exception as error:  # e.g. no internet connection; the text is still shown
                self.speech_queue.put((speech_id, None, error))

        threading.Thread(target=synthesize, daemon=True).start()

    def _stop_speaking(self, method):
        """Stop the voice. If speech was cut off by the participant (method given), it is logged as a skip."""
        skip = None
        if self.speech_pending:  # stopped before the speech even started
            skip = {"played_seconds": 0.0, "length_seconds": None}
            self.speech_pending = False
            self.speech_id += 1  # the synthesis that is still running will be ignored
        elif self.player is not None:
            skip = dm.stop_speech(self.player)
        self.player = None
        if skip and method:
            dm.log(self.log_path, {"event": "tts_skip", "turn": self.turn, "method": method, **skip})

    def _poll(self):
        """Start speech that has been synthesized and keep the audio library running."""
        try:
            while True:
                speech_id, mp3, error = self.speech_queue.get_nowait()
                if speech_id != self.speech_id or not self.speech_pending:
                    continue  # an older utterance, or the participant already stopped it
                self.speech_pending = False
                if error is None:
                    try:
                        self.player = dm.tts_play(dm.tts_audio(None, mp3))
                    except Exception as play_error:  # unreadable audio must not stop the window
                        error = play_error
                if error is not None:
                    dm.log(self.log_path, {"event": "tts_error", "turn": self.turn, "error": repr(error)})
        except queue.Empty:
            pass
        pyglet.clock.tick()
        pyglet.app.platform_event_loop.dispatch_posted_events()
        if self.args.tts:
            speaking = self.speech_pending or (self.player is not None and self.player.playing)
            self.stop_button.configure(state="normal" if speaking else "disabled")
        self.root.after(POLL_MS, self._poll)


def main(argv=None):
    """Start the window; argv are command line options (the study launchers pass their own)."""
    parser = argparse.ArgumentParser(description="Window for the restaurant recommendation dialog system.")
    parser.add_argument("--scenario", required=True, help="scenario file shown to the participant, e.g. scenarios/scenario_a.json")
    parser.add_argument("--participant", help="participant ID, written to the interaction log")
    parser.add_argument("--tts", action="store_true", help="read the system utterances aloud")
    parser.add_argument("--transparency", default="on", choices=["on", "off"],
                        help="show the reasoning behind a recommendation or only the restaurant (default: on)")
    parser.add_argument("--recommendation-mode", default="patient", choices=dm.RECOMMENDATION_MODES,
                        help="recommend after all preferences (patient, default) or after the first one (eager)")
    parser.add_argument("--matcher", default="levenshtein", choices=MATCHERS,
                        help="fallback for slot extraction when keyword matching fails (default: levenshtein)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for choosing between restaurants")
    parser.add_argument("--no-log", action="store_true", help="do not write an interaction log")
    parser.add_argument("--log-dir", default=dm.LOG_DIR, help=f"where to write interaction logs (default: {dm.LOG_DIR})")
    dm.restaurant_file_option(parser)
    args = parser.parse_args(argv)

    scenario = load_scenario(args.scenario)
    dm.apply_restaurant_file_option(args)
    model = load_model(dm.CLASSIFIER)
    if model is None:
        sys.exit(f"Error: model '{dm.CLASSIFIER}' not found, run python train.py first")
    if args.matcher == "semantic":
        warm_up()

    root = tk.Tk()
    DialogWindow(root, args, scenario, model)
    root.mainloop()


if __name__ == "__main__":
    main()
