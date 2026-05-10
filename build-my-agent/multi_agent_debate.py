import time
import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from config import get_client, get_model
from multi_agent_conversation import generate

# Setup logging
logging.basicConfig(
    filename='debate.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DebateSystem")

@dataclass
class Argument:
    """A structured debate argument."""
    debater: str
    round_num: int
    claim: str
    evidence: str
    rebuttal: str = ""
    confidence: float = 0.8

    def to_text(self):
        parts = [f"CLAIM: {self.claim}", f"EVIDENCE: {self.evidence}"]
        if self.rebuttal:
            parts.append(f"REBUTTAL: {self.rebuttal}")
        return "\n".join(parts)

    def to_dict(self):
        return asdict(self)

@dataclass
class JudgmentScore:
    """Score for a debater from the judge."""
    debater: str
    evidence_quality: float    # 0-10
    reasoning_quality: float   # 0-10
    rebuttal_quality: float    # 0-10
    overall: float = 0.0

    def compute_overall(self):
        self.overall = round(
            0.4 * self.evidence_quality +
            0.35 * self.reasoning_quality +
            0.25 * self.rebuttal_quality, 2
        )
        return self.overall

class DebateValidator:
    """Checks if an argument is valid and not just a template."""
    @staticmethod
    def is_valid(arg: Argument) -> Tuple[bool, str]:
        placeholders = [
            "<Your point>", "<Your data/logic>", "<Counter opponent>",
            "[your claim]", "[supporting evidence]", "[rebuttal]",
            "<point>", "<evidence>", "<address opponent>",
            "1-2 sentences", "2-3 sentences", "powerful sentences"
        ]
        text = (arg.claim + " " + arg.evidence + " " + arg.rebuttal).lower()
        
        for p in placeholders:
            if p.lower() in text:
                return False, f"Detected placeholder: '{p}'"
        
        if len(arg.claim) < 10:
            return False, "Claim too short."
            
        if "topic:" in arg.claim.lower() and len(arg.claim) < 50:
            return False, "Claim just regurgitates topic."
            
        return True, "Valid"

class DebaterAgent:
    """An agent that argues from a given perspective."""
    def __init__(self, name: str, perspective: str):
        self.name = name
        self.perspective = perspective
        self.system_prompt = (
            f"SYSTEM: You are {name}. Perspective: {perspective}.\n"
            "GOAL: Provide a real, persuasive argument with data and logic.\n"
            "RULES:\n"
            "1. NO placeholders. NO meta-talk. NO brackets [].\n"
            "2. Format inside <response>:\n"
            "CLAIM: <Your primary argument>\n"
            "EVIDENCE: <Detailed supporting data>\n"
            "REBUTTAL: <Directly address opponents>\n"
            "3. Wrap thought in <thought> and response in <response>."
        )
        self.arguments: List[Argument] = []

    def make_argument(self, topic: str, round_num: int, previous_arguments: List[Argument] = None, max_retries: int = 2) -> Argument:
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if round_num == 1:
            user_msg = f"Topic: {topic}\nROUND 1: Opening argument."
        else:
            others = [a for a in (previous_arguments or []) if a.debater != self.name]
            prev_text = "\n".join([f"{a.debater}: {a.claim}" for a in others])
            user_msg = f"Topic: {topic}\nPREVIOUS ROUND:\n{prev_text}\nROUND {round_num}: Respond and rebut."

        for attempt in range(max_retries + 1):
            messages.append({"role": "user", "content": user_msg})
            raw_response = generate(messages, max_tokens=400) 
            
            # Post-processing
            raw_response = re.sub(r"\[.*?\]", "", raw_response)
            
            claim = self._extract_field(raw_response, "CLAIM")
            evidence = self._extract_field(raw_response, "EVIDENCE")
            rebuttal = self._extract_field(raw_response, "REBUTTAL")

            arg = Argument(debater=self.name, round_num=round_num, claim=claim, evidence=evidence, rebuttal=rebuttal)
            
            is_valid, reason = DebateValidator.is_valid(arg)
            if is_valid:
                self.arguments.append(arg)
                logger.info(f"Agent {self.name} Round {round_num} Success: {arg.claim[:50]}...")
                return arg
            else:
                logger.warning(f"Agent {self.name} Round {round_num} Attempt {attempt} failed: {reason}")
                user_msg = f"Your previous response was invalid: {reason}. Please provide a REAL argument without placeholders or regurgitating the topic."
                # Pop the last user message to replace it or just append the correction
                messages.pop() 

        # Final fallback
        if not arg.claim or "topic:" in arg.claim.lower():
            arg.claim = f"I strongly support the perspective that {self.perspective}."
        self.arguments.append(arg)
        return arg

    def _extract_field(self, text: str, field_name: str) -> str:
        pattern = rf"{field_name}\s*:\s*(.*?)(?=(?:CLAIM|EVIDENCE|REBUTTAL)\s*:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def reset(self):
        self.arguments = []

class JudgeAgent:
    """Evaluates debate arguments."""
    def __init__(self):
        self.system_prompt = (
            "You are an impartial judge. Evaluate debaters 0-10 on EVIDENCE, REASONING, and REBUTTAL.\n"
            "Format inside <response>:\n"
            "DEBATER: [Name]\n"
            "EVIDENCE_SCORE: [Number]\n"
            "REASONING_SCORE: [Number]\n"
            "REBUTTAL_SCORE: [Number]\n"
            "...\n"
            "CONCLUSION: [Your verdict]\n"
            "WINNER: [Name]\n"
        )

    def evaluate(self, topic: str, all_arguments: List[Argument], debater_names: List[str]) -> Dict:
        transcript = ""
        for a in all_arguments:
            transcript += f"{a.debater} (R{a.round_num}):\n{a.to_text()}\n\n"

        messages = [
            {"role": "system", "content": self.system_prompt + "\nWrap in <thought> and <response>."},
            {"role": "user", "content": f"Topic: {topic}\nTranscript:\n{transcript}\n\nProvide the final judgment."}
        ]
        raw_response = generate(messages, max_tokens=1000)
        
        scores = []
        for name in debater_names:
            # Flexible parsing
            try:
                block = re.search(rf"DEBATER:\s*{re.escape(name)}.*?(?=DEBATER:|CONCLUSION:|$)", raw_response, re.S | re.I).group()
                e = float(re.search(r"EVIDENCE_SCORE:\s*([\d.]+)", block, re.I).group(1))
                r = float(re.search(r"REASONING_SCORE:\s*([\d.]+)", block, re.I).group(1))
                rb = float(re.search(r"REBUTTAL_SCORE:\s*([\d.]+)", block, re.I).group(1))
                js = JudgmentScore(name, e, r, rb)
            except:
                js = JudgmentScore(name, 5.0, 5.0, 5.0)
            js.compute_overall()
            scores.append(js)

        w_m = re.search(r"WINNER:\s*([^\n]*)", raw_response, re.I)
        c_m = re.search(r"CONCLUSION:\s*(.*?)(?=\nWINNER:|$)", raw_response, re.S | re.I)

        return {
            "scores": scores,
            "conclusion": c_m.group(1).strip() if c_m else "N/A",
            "winner": w_m.group(1).strip() if w_m else "Unknown"
        }

class DebateArena:
    def __init__(self, topic: str, num_rounds: int = 3):
        self.topic = topic
        self.num_rounds = num_rounds
        self.debaters: List[DebaterAgent] = []
        self.judge = JudgeAgent()
        self.history = []

    def add_debater(self, debater: DebaterAgent):
        self.debaters.append(debater)

    def run(self, verbose: bool = True) -> Dict:
        logger.info(f"Starting Debate: {self.topic}")
        all_args = []
        
        for r in range(1, self.num_rounds + 1):
            if verbose: print(f"\n--- ROUND {r} ---")
            for d in self.debaters:
                prev = [a for a in all_args if a.round_num == r - 1]
                arg = d.make_argument(self.topic, r, prev)
                all_args.append(arg)
                if verbose: print(f"[{d.name}] {arg.claim[:100]}...")

        judgment = self.judge.evaluate(self.topic, all_args, [d.name for d in self.debaters])
        
        if verbose:
            print(f"\n--- RESULTS ---")
            for s in judgment['scores']: print(f"{s.debater:15}: {s.overall}")
            print(f"WINNER: {judgment['winner']}")
            
        logger.info(f"Debate Finished. Winner: {judgment['winner']}")
        return judgment

if __name__ == "__main__":
    arena = DebateArena("Should AI models be open source?", num_rounds=2)
    arena.add_debater(DebaterAgent("OpenSourceAdvocate", "Transparency and democratization."))
    arena.add_debater(DebaterAgent("SafetySkeptic", "Preventing misuse by malicious actors."))
    arena.run()
