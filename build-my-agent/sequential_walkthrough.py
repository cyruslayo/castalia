from sequential_pipelines import (
    AgentNode, Pipeline, ValidatedAgentNode, 
    ConditionalPipeline, PipelineMessage, create_research_pipeline
)

def run_standard_pipeline_demo():
    print("\n--- DEMO 1: Standard Research Pipeline ---")
    pipeline = create_research_pipeline()
    # Running 2 stages to see the transformation from raw research to fact-checked text
    topic = "The impact of the James Webb Space Telescope on our understanding of the early universe."
    result = pipeline.run(topic, max_stages=2)
    result.show_flow()

def run_validation_demo():
    print("\n--- DEMO 2: Validated Agent Node ---")
    
    def length_validator(text: str) -> bool:
        # Strict validator: must be under 50 words
        word_count = len(text.split())
        return word_count <= 50

    short_summarizer = ValidatedAgentNode(
        role="short_summarizer",
        system_prompt="Summarize the input in under 30 words. BE EXTREMELY BRIEF.",
        validator=length_validator,
        temperature=0.3
    )
    
    pipeline = Pipeline("Validation Test", [short_summarizer])
    
    # This should pass if the model obeys the prompt
    input_text = "The Roman Empire was one of the largest and most influential empires in history, spanning three continents and lasting for centuries. It left a lasting legacy in law, language, and architecture."
    result = pipeline.run(input_text)
    result.show_flow()

def run_conditional_demo():
    print("\n--- DEMO 3: Conditional Routing Pipeline ---")
    
    classifier = AgentNode(
        role="classifier",
        system_prompt="""Classify the input into exactly one category: TECHNICAL or SIMPLE.
Respond with ONLY the word TECHNICAL or SIMPLE.
TECHNICAL: Science, code, complex math, or jargon.
SIMPLE: Daily life, hobbies, or general topics."""
    )
    
    tech_editor = AgentNode(
        role="tech_editor",
        system_prompt="You are a technical editor. Enhance the technical precision of the text. Keep it rigorous."
    )
    
    simple_editor = AgentNode(
        role="simple_editor",
        system_prompt="You are a casual editor. Make the text more engaging and fun for a general audience."
    )
    
    def complexity_router(msg: PipelineMessage) -> str:
        content = msg.content.upper()
        if "TECHNICAL" in content:
            return "tech_editor"
        return "simple_editor"

    cond_pipe = ConditionalPipeline("Complexity-Aware Editor")
    cond_pipe.add_stage(classifier, router=complexity_router)
    cond_pipe.add_stage(tech_editor)
    cond_pipe.add_stage(simple_editor)
    
    print("\nTest A: Technical content")
    tech_result = cond_pipe.run("Explain the mechanism of CRISPR-Cas9 in prokaryotic immune systems.")
    
    print("\nTest B: Simple content")
    simple_result = cond_pipe.run("Write a fun fact about why golden retrievers are friendly.")

    print("\nRouting Results:")
    print(f"Test A took route: {' -> '.join(m.source_agent for m in tech_result.intermediate_results)}")
    print(f"Test B took route: {' -> '.join(m.source_agent for m in simple_result.intermediate_results)}")

if __name__ == "__main__":
    run_standard_pipeline_demo()
    run_validation_demo()
    run_conditional_demo()
