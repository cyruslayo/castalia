import time
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Union
from datetime import datetime
from multi_agent_conversation import generate

@dataclass
class PipelineMessage:
    """Standard format for inter-agent communication in pipelines.
    
    Attributes:
        content: The text content of the message.
        metadata: Key-value pairs for tracking processing time, stage, etc.
        source_agent: The name of the agent that produced this message.
        timestamp: ISO format timestamp of when the message was created.
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_agent: str = "user"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self, max_len: int = 120) -> str:
        """Short preview for logging."""
        preview = self.content[:max_len].replace("\n", " ")
        if len(self.content) > max_len:
            preview += "..."
        return f"[{self.source_agent} @ {self.timestamp[11:19]}] {preview}"

    def with_metadata(self, **kwargs) -> 'PipelineMessage':
        """Return a copy with additional metadata fields."""
        new_meta = {**self.metadata, **kwargs}
        return PipelineMessage(
            content=self.content,
            metadata=new_meta,
            source_agent=self.source_agent,
            timestamp=self.timestamp,
        )

@dataclass
class AgentNode:
    """A single specialized agent in a pipeline.
    
    Attributes:
        role: A short identifier for the agent (e.g., 'researcher').
        system_prompt: Defines the agent's expertise and behavior.
        max_tokens: Controls the maximum length of the output.
        temperature: Controls the creativity/randomness of the output.
    """
    role: str
    system_prompt: str
    max_tokens: int = 700
    temperature: float = 0.7

    def process(self, input_message: PipelineMessage) -> PipelineMessage:
        """Process an input message and return an output message."""
        # We use the standardized <thought>/<response> format from multi_agent_conversation
        full_system_prompt = (
            f"{self.system_prompt}\n\n"
            "INSTRUCTION: You MUST use the following format for your response:\n"
            "<thought>\n[Your internal reasoning here]\n</thought>\n"
            "<response>\n[Your actual message to the next stage here]\n"
            "</response>"
        )
        
        messages = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": input_message.content},
        ]

        start_time = time.time()
        # Use the generate function that handles the tag extraction
        response_content = generate(
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        elapsed = time.time() - start_time

        return PipelineMessage(
            content=response_content,
            metadata={
                **input_message.metadata,
                "processing_time": round(elapsed, 2),
                "stage": self.role,
                "input_length": len(input_message.content),
                "output_length": len(response_content),
            },
            source_agent=self.role,
        )

    def __repr__(self):
        return f"AgentNode(role='{self.role}')"

@dataclass
class PipelineResult:
    """Complete result from a pipeline run."""
    final_output: Optional[PipelineMessage] = None
    intermediate_results: List[PipelineMessage] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    total_time: float = 0.0
    stages_completed: int = 0

    def show_flow(self):
        """Print the flow of messages through the pipeline."""
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION FLOW")
        print("=" * 70)
        for i, msg in enumerate(self.intermediate_results):
            stage_label = f"Stage {i}" if i > 0 else "Input"
            print(f"\n{'-' * 70}")
            print(f"  {stage_label}: [{msg.source_agent}]")
            if 'processing_time' in msg.metadata:
                print(f"  Time: {msg.metadata['processing_time']}s | "
                      f"Output length: {msg.metadata.get('output_length', 'N/A')} chars")
            print(f"{'-' * 70}")
            print(msg.content[:500])
            if len(msg.content) > 500:
                print(f"  ... ({len(msg.content) - 500} more chars)")
        
        if self.errors:
            print(f"\n[!] Errors: {len(self.errors)}")
            for e in self.errors:
                print(f"  - Stage '{e['stage']}': {e['error']}")
        
        print(f"\n{'=' * 70}")
        print(f"Total time: {self.total_time:.2f}s | Stages completed: {self.stages_completed}")
        print("=" * 70 + "\n")

class Pipeline:
    """Sequential pipeline that chains AgentNodes."""

    def __init__(self, name: str, stages: List[AgentNode]):
        self.name = name
        self.stages = stages

    def run(
        self,
        input_message: Union[str, PipelineMessage],
        max_stages: Optional[int] = None,
        max_retries: int = 1,
    ) -> PipelineResult:
        """Execute the pipeline."""
        if isinstance(input_message, str):
            input_message = PipelineMessage(content=input_message)
            
        result = PipelineResult()
        result.intermediate_results.append(input_message)
        current = input_message
        stages_to_run = self.stages[:max_stages] if max_stages else self.stages
        pipeline_start = time.time()

        print(f"PIPELINE: '{self.name}' starting ({len(stages_to_run)} stages)")
        for i, stage in enumerate(stages_to_run):
            print(f"  [Stage {i + 1}/{len(stages_to_run)}]: {stage.role}...", end=" ", flush=True)
            attempts = 0
            success = False

            while attempts < max_retries and not success:
                try:
                    current = stage.process(current)
                    result.intermediate_results.append(current)
                    result.stages_completed += 1
                    success = True
                    print(f"DONE ({current.metadata.get('processing_time', '?')}s)")
                except Exception as e:
                    attempts += 1
                    if attempts >= max_retries:
                        error_info = {"stage": stage.role, "error": str(e), "attempt": attempts}
                        result.errors.append(error_info)
                        print(f"FAIL (failed after {attempts} attempt(s))")
                        break
                    print(f"RETRY {attempts}...", end=" ", flush=True)

        result.total_time = time.time() - pipeline_start
        result.final_output = current
        print(f"PIPELINE COMPLETE in {result.total_time:.2f}s")
        return result

    def __repr__(self):
        stage_names = " -> ".join(s.role for s in self.stages)
        return f"Pipeline('{self.name}': {stage_names})"

class ValidatedAgentNode(AgentNode):
    """AgentNode with output validation."""

    def __init__(self, role: str, system_prompt: str, validator: Callable[[str], bool] = None, **kwargs):
        super().__init__(role=role, system_prompt=system_prompt, **kwargs)
        self.validator = validator

    def process(self, input_message: PipelineMessage) -> PipelineMessage:
        result = super().process(input_message)
        if self.validator and not self.validator(result.content):
            raise ValueError(
                f"Validation failed for stage '{self.role}': output did not pass quality check"
            )
        return result

class ConditionalPipeline:
    """Pipeline with conditional branching based on stage output."""

    def __init__(self, name: str):
        self.name = name
        self.stages: Dict[str, AgentNode] = {}
        self.routes: Dict[str, Callable[[PipelineMessage], Optional[str]]] = {}
        self.start_stage: Optional[str] = None

    def add_stage(self, agent: AgentNode, next_stage: str = None,
                  router: Callable[[PipelineMessage], str] = None):
        """Add a stage with either a fixed next stage or a dynamic router."""
        self.stages[agent.role] = agent
        if not self.start_stage:
            self.start_stage = agent.role
        if router:
            self.routes[agent.role] = router
        elif next_stage:
            self.routes[agent.role] = lambda msg, ns=next_stage: ns

    def run(self, input_message: Union[str, PipelineMessage], max_steps: int = 10) -> PipelineResult:
        """Execute the conditional pipeline."""
        if isinstance(input_message, str):
            input_message = PipelineMessage(content=input_message)
            
        result = PipelineResult()
        result.intermediate_results.append(input_message)
        current = input_message
        current_stage_name = self.start_stage
        pipeline_start = time.time()

        print(f"CONDITIONAL PIPELINE: '{self.name}' starting")
        step = 0
        while current_stage_name and step < max_steps:
            stage = self.stages.get(current_stage_name)
            if not stage:
                print(f"  WARNING: Stage '{current_stage_name}' not found — stopping")
                break

            print(f"  [Step {step + 1}]: {stage.role}...", end=" ", flush=True)
            try:
                current = stage.process(current)
                result.intermediate_results.append(current)
                result.stages_completed += 1
                print("DONE")
            except Exception as e:
                result.errors.append({"stage": stage.role, "error": str(e)})
                print(f"FAIL ({e})")
                break

            # Determine next stage
            router = self.routes.get(current_stage_name)
            current_stage_name = router(current) if router else None
            step += 1

        result.total_time = time.time() - pipeline_start
        result.final_output = current
        print(f"DONE in {result.total_time:.2f}s ({result.stages_completed} stages)")
        return result

def create_research_pipeline() -> Pipeline:
    """Creates a standard 3-stage research pipeline."""
    researcher = AgentNode(
        role="researcher",
        system_prompt="""You are a thorough research analyst. Your job:
1. Gather comprehensive information on the given topic.
2. Cover multiple angles and perspectives.
3. Include specific facts, figures, and examples where possible.
4. Organize information clearly with headers and bullet points.
Output ONLY your research findings. Do not summarize or conclude."""
    )
    
    fact_checker = AgentNode(
        role="fact_checker",
        system_prompt="""You are a meticulous fact-checker. Your job:
1. Read the research text provided.
2. Identify factual claims and assess if they are VERIFIED, PLAUSIBLE, or QUESTIONABLE.
3. Flag any contradictions or logical errors.
4. Output the CORRECTED version of the research.
Format: Start with 'FACT-CHECK SUMMARY:' then provide 'CORRECTED TEXT:'.""",
        temperature=0.2
    )
    
    summarizer = AgentNode(
        role="summarizer",
        system_prompt="""You are an expert summarizer. Your job:
1. Distill the fact-checked research to 3-5 key points.
2. Preserve important facts and figures.
3. End with a one-sentence takeaway.
Keep the summary under 200 words.""",
        temperature=0.5
    )
    
    return Pipeline("Standard Research Pipeline", [researcher, fact_checker, summarizer])

if __name__ == "__main__":
    # Smoke test
    pipe = create_research_pipeline()
    test_input = "What are the primary factors contributing to the melting of Arctic sea ice?"
    # Just run the first stage for a quick check if needed, but let's try the full pipe
    result = pipe.run(test_input, max_stages=1) # Limiting to 1 stage for the smoke test to save time/tokens
    result.show_flow()
