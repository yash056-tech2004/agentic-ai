from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

try:
    from report_formatter import assemble_final_report, DEFAULT_AUTHOR
except ImportError:
    # Supports imports when main.py is loaded as src.main from frontend apps.
    from src.report_formatter import assemble_final_report, DEFAULT_AUTHOR


# ----- Helpers ---------------------------------------------------------------

def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s-]", "", value)
    value = re.sub(r"[\s-]+", "-", value)
    return value[:80] if value else "topic"


def get_llm(provider: str, model: str, temperature: float):
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature)

    raise ValueError("Unsupported provider. Use 'openai', 'anthropic', or 'groq'.")


def build_tools() -> list[Tool]:
    web_tool = DuckDuckGoSearchRun(name="web_search")

    wiki_api = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2500)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)

    return [
        Tool(
            name="WebSearch",
            func=web_tool.run,
            description="Search the open web for recent, high-signal context.",
        ),
        Tool(
            name="WikipediaKnowledge",
            func=wiki_tool.run,
            description="Fetch reliable encyclopedic context from Wikipedia.",
        ),
    ]


def build_react_prompt() -> PromptTemplate:
    template = """
You are an Autonomous Research Agent using ReAct-style reasoning.

You must:
1) Investigate the user topic with the available tools.
2) Cross-check and avoid single-source claims.
3) Deliver a concise, well-structured markdown report.

Available tools:
{tools}

Use this exact reasoning format:
Question: the input topic
Thought: your reasoning step
Action: one of [{tool_names}]
Action Input: the search query sent to the tool
Observation: result returned by the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information
Final Answer: the final markdown report only

The Final Answer MUST include these sections in order:
# Title
## Introduction
## Key Findings
## Challenges
## Future Scope
## Conclusion

Be crisp, cite concrete facts, and keep the tone neutral-professional.
""".strip()

    return PromptTemplate.from_template(template)


def create_agent_executor(provider: str, model: str, temperature: float) -> AgentExecutor:
    llm = get_llm(provider, model, temperature)
    tools = build_tools()
    prompt = build_react_prompt()

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
    )


def run_research(topic: str, provider: str, model: str, temperature: float) -> str:
    executor = create_agent_executor(provider, model, temperature)
    result = executor.invoke({"input": topic})

    output = result.get("output")
    if not output:
        raise RuntimeError("Agent did not produce an output.")

    return assemble_final_report(topic=topic, report_body=output, author=DEFAULT_AUTHOR)


def save_report(topic: str, report: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{slugify(topic)}_{ts}.md"
    output_path = out_dir / filename
    output_path.write_text(report, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Research Agent (LangChain)")
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument(
        "--provider",
        default="groq",
        choices=["openai", "anthropic", "groq"],
        help="LLM provider",
    )
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Model name. Example: gpt-4o-mini, claude-3-5-sonnet-latest, llama-3.3-70b-versatile",
    )
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    # Fast fail with clear messaging for missing keys
    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set. Add it to .env")
    if args.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Add it to .env")
    if args.provider == "groq" and not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY not set. Add it to .env")

    report = run_research(
        topic=args.topic,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )

    output_path = save_report(args.topic, report, Path(args.out_dir))
    print("\n=== FINAL REPORT GENERATED ===")
    print(report)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
