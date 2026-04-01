# Assignment 2: Autonomous Research Agent

A LangChain-based autonomous research agent that searches the web, extracts knowledge, and generates a structured research report. It supports multiple LLM providers and includes a Streamlit UI plus a CLI. Authored for Ravi Deo (System ID: 2023458249).

## Features

- ReAct-style agent with tool use (web search + Wikipedia)
- Supports OpenAI, Anthropic, and Groq providers
- Outputs consistent markdown reports with enforced sections
- Streamlit UI for one-click report generation
- CLI interface for automation and testing
- Optional PDF export via `generate_report_pdf.py`

## Tech Stack

- Python 3.12+
- LangChain + langchain_classic (ReAct agent)
- DuckDuckGo Search + Wikipedia tools
- Streamlit UI
- ReportLab for PDF rendering

## Project Structure

```
Assignment 2/
├─ frontend.py
├─ src/
│  ├─ main.py
│  └─ report_formatter.py
├─ outputs/
├─ docs/
│  └─ Project_Report_Assignment_2.pdf (generated)
├─ samples/
│  ├─ sample_output_1.md
│  └─ sample_output_2.md
├─ .env.example (to create)
├─ requirements.txt
└─ README.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and set your API key(s).

## Environment Variables

Set the provider(s) you plan to use:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...
```

## Run (Frontend)

```bash
streamlit run frontend.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Run (CLI)

### OpenAI

```bash
python src/main.py --topic "Impact of AI in Healthcare" --provider openai --model gpt-4o-mini
```

### Anthropic

```bash
python src/main.py --topic "Impact of AI in Healthcare" --provider anthropic --model claude-3-5-sonnet-latest
```

### Groq

```bash
python src/main.py --topic "Impact of AI in Healthcare" --provider groq --model llama-3.3-70b-versatile
```

## Output Format

Reports are saved as markdown in `outputs/` and include:

- Cover Page (with author Ravi Deo, System ID: 2023458249)
- Title
- Introduction
- Key Findings
- Challenges
- Future Scope
- Conclusion

## Sample Outputs

Sample reports are stored in `samples/`.

## Troubleshooting

- If a provider fails, verify the correct API key is set in `.env`.
- If the UI is not loading, re-run `streamlit run frontend.py`.
- If you see tool errors, reinstall dependencies from `requirements.txt`.
- For PDF generation, run `python generate_report_pdf.py` to write the file into `docs/`.

## Submission Notes

- Upload the full repository to GitHub.
- Include sample reports from `samples/` and the generated PDF in `docs/`.
