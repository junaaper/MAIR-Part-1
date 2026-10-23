# Study condition: scenario A (romantic dinner) with text-to-speech
# Usage: python scenario_a_tts.py --participant P01
import os
import sys

import ui

SCENARIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios", "scenario_a.json")

if __name__ == "__main__":
    ui.main(["--scenario", SCENARIO, "--tts"] + sys.argv[1:])
