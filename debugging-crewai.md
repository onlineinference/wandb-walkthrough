# Debugging CrewAI Multi-Agent Applications

This article explores combining **CrewAI's** multi-agent orchestration with **W&B Weave's** observability for debugging complex AI workflows. We'll use a log analysis and debugging agent as an example.

---

## Understanding CrewAI and its Capabilities

**CrewAI** facilitates building multi-agent AI systems by allowing you to define specialized agents, assign tasks, and coordinate their collaboration. This modular approach improves reliability, transparency, and extensibility.

### Key features of CrewAI

CrewAI enables defining agents with roles, boundaries, and tools, managing tasks with dependencies, and integrating external APIs. It supports both Python code and YAML configurations for agent and workflow definitions.

---

## Introducing W&B Weave for Enhanced Observability

**W&B Weave** provides observability for multi-agent systems, integrating directly with CrewAI to capture and log agent activity. This allows visualization of agent processes, decision paths, and information flow for debugging and optimization.

---

## Monitoring Cost, Token Usage, and Agent Failures

Weave offers tools to monitor operational metrics like costs, token usage, latency, and agent failures through dashboards. This transparency helps identify cost drivers, optimize resource use, detect bottlenecks, and quickly diagnose issues.

---

## Tutorial: Building a Code Debugger Agent with CrewAI

This tutorial builds a Python code debugger agent using **CrewAI** and **Weave**. The agent analyzes stderr output, identifies root causes, generates search queries, finds solutions on the web and GitHub, and produces an HTML debug report.

---

### Step 1: Creating a Logging System Using a Bash Alias

This step creates a Bash function, `agentpython`, to capture Python script errors into `/tmp/agentpython-stderr.log`. If errors exist, the log is passed to the debugging agent.

```bash
agentpython() {
    logfile="/tmp/agentpython-stderr.log"
    python "$@" 2> >(tee "$logfile" >&2)
    if [[ -s "$logfile" ]]; then
        # If logfile is NOT empty, run check script
        python /Users/brettyoung/Desktop/dev25/tutorials/dbg_crw/debug_main.py "$logfile"
    else
        # If logfile is empty, clear it (truncate to zero length)
        > "$logfile"
    fi
}
````

To add this alias, run:

```bash
profile_file=$(test -f ~/.zshrc && echo ~/.zshrc || (test -f ~/.bashrc && echo ~/.bashrc)); echo 'agentpython() {
    logfile="/tmp/agentpython-stderr.log"
    python "$@" 2> >(tee "$logfile" >&2)
    if [[ -s "$logfile" ]]; then
        python full_path_to_your_script "$logfile"
    else
        > "$logfile"
    fi
}' >> "$profile_file" && source "$profile_file" && echo "Added and sourced $profile_file"
```

-----

### Step 2: Creating a "buggy" script to test with

Create a Python script, `bad_code.py`, that causes a NumPy error for testing the debugging agent:

```python
import numpy as np

# Create a structured array
dt = np.dtype([('x', 'f8'), ('y', 'i4')])
arr = np.zeros(100, dtype=dt)

# Fill with data
arr['x'] = np.random.random(100)
arr['y'] = np.arange(100)

# Create problematic buffer view
buffer_data = arr.tobytes()[:-5]  # Truncated buffer

# This triggers a numpy buffer/memory bug
corrupted = np.frombuffer(buffer_data, dtype=np.complex128, count=-1)

# Try to use the corrupted array - this often segfaults
result = np.fft.fft(corrupted) * np.ones(len(corrupted))
print(f"Result shape: {result.shape}")
```

-----

### Step 3: Building Out Our Agent with CrewAI and Weave

This section outlines the debugging agent's implementation using **CrewAI** and **Weave**. It loads the stderr log, analyzes the error, generates search queries, searches GitHub and the web, reviews code snippets, and compiles a debug report.

The code defines tool input schemas, custom tools for log reading, search query generation, combined GitHub/web search, file analysis, file snippet extraction, tool suggestions, and report generation. It then sets up CrewAI agents (`Log Analysis Specialist`, `Search Query Specialist`, `Combined Repository and Web Search Specialist`, `Code Analysis Specialist`, `Debug Report Generator`) and orchestrates them through a sequence of tasks to automate the debugging workflow.

````python
import os
import sys
import re
import requests
import tempfile
import webbrowser
import html
from pathlib import Path
from typing import Type, List, Optional, Dict, Any, Union
import json 

from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai import BaseLLM, LLM
import weave; weave.init("crewai_debug_agent")

from langchain_openai import ChatOpenAI
import os
import re
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Type
import subprocess


LOGFILE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/agentpython-stderr.log"
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

# Read the log file BEFORE kicking off
LOG_CONTENT = ""
if os.path.exists(LOGFILE) and os.path.getsize(LOGFILE) > 0:
    with open(LOGFILE, 'r') as f:
        LOG_CONTENT = f.read()
    print(f"\033[95m[LOG CONTENT LOADED] {len(LOG_CONTENT)} characters from {LOGFILE}\033[0m")
else:
    LOG_CONTENT = "No log file found or file is empty"
    print(f"\033[95m[LOG] No content found in {LOGFILE}\033[0m")

def verbose_print(msg):
    print(f"\033[95m[LOG] {msg}\033[0m", flush=True)



# ----- Tool Input Schemas -----
class LogAnalysisInput(BaseModel):
    log_content: str = Field(..., description="Log content to analyze (already loaded)")

class SearchQueryInput(BaseModel):
    error_text: str = Field(..., description="Error text to generate search query from")

class CombinedSearchInput(BaseModel):
    query: str = Field(..., description="Search query for both GitHub issues and web")
    owner: str = Field(default="", description="GitHub repository owner")
    repo: str = Field(default="", description="GitHub repository name")

class FileAnalysisInput(BaseModel):
    log_content: str = Field(..., description="Log content to extract file information from")

class FileSnippetInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to get snippet from")
    line: Optional[int] = Field(default=None, description="Line number to focus on")
    n_lines: int = Field(default=20, description="Number of lines to return")

class ToolSuggestionInput(BaseModel):
    error_message: str = Field(..., description="Error message to analyze")
    code_snippet: str = Field(..., description="Code snippet related to the error")

class ReportGenerationInput(BaseModel):
    log: str = Field(..., description="Error log content")
    file_snippet: str = Field(default="", description="Relevant code snippet")
    tools: str = Field(default="", description="Tool recommendations")
    gh_results: str = Field(default="", description="GitHub search results")
    web_results: str = Field(default="", description="Web search results")

# ----- Tools -----

class LogReaderTool(BaseTool):
    name: str = Field(default="Log Reader")
    description: str = Field(default="Provides access to the pre-loaded log content")
    args_schema: Type[BaseModel] = LogAnalysisInput
    
    def _run(self, log_content: str = None) -> str:
        verbose_print(f"Using pre-loaded log content")
        
        if not LOG_CONTENT or LOG_CONTENT == "No log file found or file is empty":
            return "[LOG] Log file empty or not found. No action needed."
        
        is_python_error = "Traceback" in LOG_CONTENT or "Exception" in LOG_CONTENT or "Error" in LOG_CONTENT
        error_type = "Python Error" if is_python_error else "General Error"
        return f"Error Type: {error_type}\n\nLog Content:\n{LOG_CONTENT}"

class SearchQueryGeneratorTool(BaseTool):
    name: str = Field(default="Search Query Generator")
    description: str = Field(default="Generates optimized search queries from error messages")
    args_schema: Type[BaseModel] = SearchQueryInput
    
    def _run(self, error_text: str) -> str:
        verbose_print("Generating search query via LLM...")
        try:
            prompt = (
                "Given this error or question, write a concise search query to help the person find a solution online. "
                "Output only the query (no explanation):\n\n" + error_text
            )
            query = llm.call(prompt)
            return f"Generated search query: {query.strip()}"
        except Exception as e:
            return f"Error generating search query: {str(e)}"




class CombinedSearchTool(BaseTool):
    name: str = Field(default="Combined GitHub & Web Search")
    description: str = Field(default="Searches both GitHub issues and the web in one call, returning both results.")
    args_schema: Type[BaseModel] = CombinedSearchInput

    def _run(self, query: str, owner: str = "", repo: str = "") -> dict:
        github_results = self._github_search(query, owner, repo)
        web_results = self._web_search(query)
        return {
            "github_issues": github_results,
            "web_search": web_results
        }

    def _github_search(self, query: str, owner: str, repo: str):
        import httpx
        GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
        url = '[https://api.github.com/search/issues](https://api.github.com/search/issues)'
        headers = {'Accept': 'application/vnd.github.v3+json'}
        if GITHUB_TOKEN:
            headers['Authorization'] = f'token {GITHUB_TOKEN}'
        gh_query = f'repo:{owner}/{repo} is:issue {query}' if owner and repo else query
        params = {'q': gh_query, 'per_page': 5}
        try:
            with httpx.Client(timeout=15) as client:
                resp = client.get(url, headers=headers, params=params)
                if resp.status_code == 200:
                    items = resp.json().get("items", [])
                    return [
                        {
                            "number": item.get("number"),
                            "title": item.get("title"),
                            "url": item.get("html_url"),
                            "body": (item.get("body") or "")[:500]
                        }
                        for item in items
                    ]
                else:
                    return [{"error": f"GitHub search failed: {resp.status_code} {resp.text}"}]
        except Exception as e:
            return [{"error": f"Error searching GitHub: {str(e)}"}]

    def _extract_json(self, text):
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if not m:
            m = re.search(r"```(.*?)```", text, re.DOTALL)
        block = m.group(1) if m else text
        try:
            j = json.loads(block)
            return j if isinstance(j, list) else [j]
        except Exception:
            return []
        
    def _web_search(self, query: str, n_results: int = 5):
        # Your actual OpenAI-based tool call here
        from openai import OpenAI  # or however your actual OpenAI client is imported
        client = OpenAI()
        prompt = (
            f"Show me {n_results} of the most important/useful web results for this search along with a summary of the problem and proposed solution: '{query}'. "
            "Return as markdown JSON:\n"
            "[{\"title\": ..., \"url\": ..., \"date_published\": ..., \"snippet\": ...}]"
        )
        
        response = client.responses.create(
            model="gpt-4.1",  # or "gpt-4.1", or your available web-enabled model
            tools=[{"type": "web_search_preview"}],
            input=prompt,
        )
        return self._extract_json(response.output_text)




class FileAnalysisTool(BaseTool):
    name: str = Field(default="File Analysis")
    description: str = Field(default="Extracts file paths and line numbers from error logs")
    args_schema: Type[BaseModel] = FileAnalysisInput
    
    def _run(self, log_content: str = None) -> str:
        verbose_print("Invoking LLM to identify files from log...")
        # Use the global LOG_CONTENT if log_content not provided
        content_to_analyze = log_content or LOG_CONTENT
        try:
            prompt = (
                "Given this error message or traceback, list all file paths (and, if available, line numbers) involved in the error. "
                "Output one JSON per line, as:\n"
                '{"file": "path/to/file.py", "line": 123}\n'
                'If line is not found, use null.\n'
                f"\nError:\n{content_to_analyze}"
            )
            output = llm.call(prompt)
            results = []
            for l in output.splitlines():
                l = l.strip()
                if not l:
                    continue
                try:
                    results.append(eval(l, {"null": None}))
                except Exception as exc:
                    verbose_print(f"[File Extraction Skipped Line]: {l!r} ({exc})")
            return f"Files found in error: {results}"
        except Exception as e:
            return f"Error analyzing files: {str(e)}"

class FileSnippetTool(BaseTool):
    name: str = Field(default="File Snippet Extractor")
    description: str = Field(default="Extracts code snippets from files around specific lines")
    args_schema: Type[BaseModel] = FileSnippetInput
    
    def _run(self, file_path: str, line: Optional[int] = None, n_lines: int = 20) -> str:
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
            if line and 1 <= line <= len(lines):
                s = max(0, line-6)
                e = min(len(lines), line+5)
                code = lines[s:e]
            else:
                code = lines[:n_lines]
            return f"Code snippet from {file_path}:\n{''.join(code)}"
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"

class ToolSuggestionTool(BaseTool):
    name: str = Field(default="Tool Suggestion")
    description: str = Field(default="Suggests which debugging tools to use next based on error analysis")
    args_schema: Type[BaseModel] = ToolSuggestionInput
    
    def _run(self, error_message: str, code_snippet: str) -> str:
        verbose_print("Requesting tool suggestions via LLM...")
        prompt = (
            "You are an AI debugging orchestrator. The following is a Python error message and a snippet of code "
            "from a file involved in the error. Based on this, choose which tools should be used next, and explain why. "
            "Possible tools: github_issue_search, web_search. "
            "Always recommend github_issue_search as it's very helpful. "
            "Provide your recommendation in a clear, structured format.\n"
            "Error:\n" + error_message + "\n\nFile snippet:\n" + code_snippet
        )
        try:
            return llm.call(prompt).strip()
        except Exception as e:
            return f"Error generating tool suggestions: {str(e)}"

class ReportGeneratorTool(BaseTool):
    name: str = Field(default="HTML Report Generator")
    description: str = Field(default="Generates HTML debug reports")
    args_schema: Type[BaseModel] = ReportGenerationInput
    
    def _run(self, log: str, file_snippet: str = "", tools: str = "", gh_results: str = "", web_results: str = "") -> str:
        verbose_print("Writing HTML report ...")
        out_path = os.path.join(tempfile.gettempdir(), 'dbg_report.html')
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("<html><head><meta charset='utf-8'><title>Debug Results</title></head><body>\n")
                f.write("<h1 style='color:#444;'>Debugging Session Report</h1>\n")
                f.write("<h2>Error Log</h2>")
                f.write("<pre style='background:#f3f3f3;padding:8px;'>" + html.escape(log or "None") + "</pre>")
                if file_snippet:
                    f.write("<h2>Relevant Source Snippet</h2><pre style='background:#fafaff;padding:8px;'>" + html.escape(file_snippet) + "</pre>")
                if tools:
                    f.write("<h2>LLM Tool Recommendations</h2><pre style='background:#eef;'>" + html.escape(tools) + "</pre>")
                if gh_results:
                    f.write("<h2>GitHub & Web Search Results</h2><pre>" + html.escape(gh_results) + "</pre>")
                if web_results:
                    f.write("<h2>Web Search AI Answer</h2><pre>" + html.escape(web_results) + "</pre>")
                f.write("</body></html>")
            return f"HTML report generated and opened at: {out_path}"
        except Exception as e:
            return f"Error generating HTML report: {str(e)}"

# --- Tool Instances
log_reader_tool = LogReaderTool()
search_query_generator_tool = SearchQueryGeneratorTool()
combined_search_tool = CombinedSearchTool()
file_analysis_tool = FileAnalysisTool()
file_snippet_tool = FileSnippetTool()
tool_suggestion_tool = ToolSuggestionTool()
report_generator_tool = ReportGeneratorTool()


class CustomChatOpenAI(ChatOpenAI):
    def call(self, prompt, system_message=None):
        """
        Run inference on a prompt (string). Optionally provide a system message.
        
        Args:
            prompt (str): The user's message.
            system_message (str, optional): The system context for the assistant.
        
        Returns:
            str: The model's response content.
        """
        messages = []
        if system_message:
            messages.append(("system", system_message))
        messages.append(("human", prompt))
        result = self.invoke(messages)
        return result.content


llm = CustomChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
fouro_llm = CustomChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)



# --- Agents ---
log_analyst_agent = Agent(
    role="Log Analysis Specialist",
    goal="Analyze the pre-loaded log content to identify errors and extract relevant information. Start by reading log with the log_reader_tool, then move on to usign the file_alaysis_tool to read important info from the file(s) involved in the error",
    backstory="Expert in parsing error logs and identifying the root causes of issues",
    tools=[log_reader_tool, file_analysis_tool],
    allow_delegation=False,
    llm=fouro_llm
)

search_specialist_agent = Agent(
    role="Search Query Specialist",
    goal="Generate optimized search queries from error messages for effective problem resolution. The search must be less that 100 chars long!!!!!!!!!!",
    backstory="Expert in crafting search queries that yield the most relevant debugging results. The search must be less that 100 chars long!!!!!!!!!!",
    tools=[search_query_generator_tool],
    allow_delegation=False,
    llm=fouro_llm
)


combined_research_agent = Agent(
    role="Combined Repository and Web Search Specialist",
    goal="Search both GitHub issues and the web for relevant solutions to errors and problems. You must use the combined_search_tool no matter what!!!!!!! Try to summarize each specific github/web problem and solution to help the user solve their issue. Make sure to include the links from the original sources next to their corresponding summaries / code etc",
    backstory="Expert in both GitHub open-source research and web documentation sleuthing for code solutions.",
    tools=[combined_search_tool],
    allow_delegation=False,
    llm=fouro_llm
)

code_analyst_agent = Agent(
    role="Code Analysis Specialist",
    goal="Analyze code snippets and suggest debugging approaches",
    backstory="Expert in code analysis and debugging strategy recommendation",
    tools=[file_snippet_tool],
    allow_delegation=False,
    llm=fouro_llm
)

report_generator_agent = Agent(
    role="Debug Report Generator",
    goal="Compile all debugging information into comprehensive HTML reports. Make sure to include the links to sources when they are provides -- but DO NOT make up links if they are not given.  Write an extensive report covering all possible solutions to the problem!!!",
    backstory="Specialist in creating detailed, actionable debugging reports",
    tools=[report_generator_tool],
    allow_delegation=False,
    llm=llm
)

# --- Tasks ---
log_analysis_task = Task(
    description=f"Analyze the pre-loaded log content. The log content is already available: {LOG_CONTENT[:500]}... Extract error information and identify the type of error.",
    expected_output="Detailed analysis of the log content including error type and content",
    agent=log_analyst_agent,
    output_file="log_analysis.md"
)

file_extraction_task = Task(
    description="Extract file paths and line numbers from the analyzed log content. Use the pre-loaded log content to identify which files are involved in the error.",
    expected_output="List of files and line numbers involved in the error",
    agent=log_analyst_agent,
    context=[log_analysis_task],
    output_file="file_analysis.md"
)

search_query_task = Task(
    description="Generate optimized search queries based on the error analysis for finding solutions online. The search must be less that 100 chars long!!!!!!!!!!",
    expected_output="Optimized search queries for the identified errors. The search must be less that 100 chars long!!!!!!!!!!",
    agent=search_specialist_agent,
    context=[log_analysis_task],
    output_file="search_queries.md"
)

combined_search_task = Task(
    description="Use the search queries to search both GitHub issues and the wide web for solutions. Make sure to make a very robust report incorporating ALL sources. Dont just give desciptions of the issue- write a detailed summary showcasing code and exact explanations to issues in the report.",
    expected_output="Relevant GitHub issues and web documentation/articles/answers.",
    agent=combined_research_agent,
    context=[search_query_task],
    output_file="combined_results.md"
)

code_analysis_task = Task(
    description="Extract and analyze code snippets from the implicated files. Suggest debugging tools and approaches.",
    expected_output="Code snippets and debugging tool recommendations",
    agent=code_analyst_agent,
    context=[file_extraction_task],
    output_file="code_analysis.md"
)

report_generation_task = Task(
    description="Compile all debugging information into a comprehensive HTML report and open it in the browser. Make sure to make a very robust report incorporating ALL sources Make sure to include the links to sources when they are provides -- but DO NOT make up links if they are not given. -- ALL sourced information must be cited!!!!!! Write an extensive report covering all possible solutions to the problem!!!",
    expected_output="Complete HTML debugging report",
    agent=report_generator_agent,
    context=[log_analysis_task, combined_search_task, code_analysis_task],
    output_file="debug_report.html"
)

# --- Run Crew ---
crew = Crew(
    agents=[
        log_analyst_agent,
        search_specialist_agent,
        combined_research_agent,
        code_analyst_agent,
        report_generator_agent
    ],
    tasks=[
        log_analysis_task,
        file_extraction_task,
        search_query_task,
        combined_search_task,
        code_analysis_task,
        report_generation_task
    ],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print(f"\033[95m[STARTING] Log content loaded: {len(LOG_CONTENT)} chars\033[0m")
    result = crew.kickoff()
    print("\n\nDebug Analysis Complete:\n")
    print(result)
    
    # Try to open the generated report
    report_path = './debug_report.html'
    if os.path.exists(report_path):
        verbose_print(f"Opening final report: {report_path}")
        if sys.platform.startswith("darwin"):
            subprocess.Popen(['open', report_path])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(['xdg-open', report_path])
        elif sys.platform.startswith("win"):
            os.startfile(report_path)
````

**Debugging with Weave:** Weave traces revealed a `ChatOpenAI` `.call` method issue. The fix was to subclass `ChatOpenAI` and add a `.call` method that wraps `.invoke()`:

```python
class CustomChatOpenAI(ChatOpenAI):
    def call(self, prompt, system_message=None):
        """
        Run inference on a prompt (string). Optionally provide a system message.
        
        Args:
            prompt (str): The user's message.
            system_message (str, optional): The system context for the assistant.
        
        Returns:
            str: The model's response content.
        """
        messages = []
        if system_message:
            messages.append(("system", system_message))
        messages.append(("human", prompt))
        result = self.invoke(messages)
        return result.content
```

**Latency Optimization:** High latency (2 minutes per run) was observed via Weave. To address this, **OpenRouter** and **Cerebras's Qwen 3 32B** were used, proxied through a FastAPI application to maintain OpenAI compatibility.

**FastAPI Proxy for OpenRouter:** This `FastAPI` application acts as a proxy, forwarding OpenAI-style chat completion requests to the OpenRouter API, specifically targeting the Cerebras-hosted Qwen model.

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import os
from typing import List, Dict, Any, Optional, Union
import requests

# Your re-used settings
API_KEY = os.getenv("OPENROUTER_API_KEY") or "your_api_key"
SITE_URL = "[https://your-site-url.com](https://your-site-url.com)"
SITE_NAME = "Your Site Name"
MODEL = "qwen/qwen3-32b"

# Your OpenRouterCerebrasLLM class (truncated for brevity; copy your full code here)
class OpenRouterCerebrasLLM:
    def __init__(
        self,
        model: str,
        api_key: str,
        site_url: str,
        site_name: str,
        temperature: Optional[float] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.endpoint = "[https://openrouter.ai/api/v1/chat/completions](https://openrouter.ai/api/v1/chat/completions)"
        
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "provider": {
                "order": ["cerebras"],
                "allow_fallbacks": False
            }
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json"
        }
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result

# Initialize the FastAPI app and the LLM once
app = FastAPI()
llm = OpenRouterCerebrasLLM(
    model=MODEL,
    api_key=API_KEY,
    site_url=SITE_URL,
    site_name=SITE_NAME,
    temperature=0.7
)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages")
    # you can also handle tools, temperature, etc. here
    try:
        raw_response = llm.call(messages)
        # This returns the full OpenRouter response. If you want to narrow down to only the OpenAI-compatible fields,
        # you could filter here, but for maximum compatibility just return as-is.
        return JSONResponse(raw_response)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    print(f"Serving local OpenAI-compatible LLM proxy on http://localhost:8001/v1/chat/completions")
    print(f"Forwarding all requests to: {MODEL}, via OpenRouter w/ your secret/settings")
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

**Agent Script with New Model:** This updated agent script replaces most OpenAI model calls with calls to the Cerebras-hosted Qwen model via the local proxy.

````python
import os
import sys
import re
import requests
import tempfile
import webbrowser
import html
from pathlib import Path
from typing import Type, List, Optional, Dict, Any, Union
import json 

from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai import LLM
import weave; weave.init("crewai_debug_agent")

from langchain_openai import ChatOpenAI
import os
import re
import asyncio
import httpx
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Type
import subprocess


LOGFILE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/agentpython-stderr.log"
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

# Read the log file BEFORE kicking off
LOG_CONTENT = ""
if os.path.exists(LOGFILE) and os.path.getsize(LOGFILE) > 0:
    with open(LOGFILE, 'r') as f:
        LOG_CONTENT = f.read()
    print(f"\033[95m[LOG CONTENT LOADED] {len(LOG_CONTENT)} characters from {LOGFILE}\033[0m")
else:
    LOG_CONTENT = "No log file found or file is empty"
    print(f"\033[95m[LOG] No content found in {LOGFILE}\033[0m")

def verbose_print(msg):
    print(f"\033[95m[LOG] {msg}\033[0m", flush=True)

# ----- LLM (local or OpenRouter, as per your local config) -----
cerebras_llm = LLM(
    model="openrouter/meta-llama/llama-4-scout",
    base_url="http://localhost:8001/v1",
    api_key="put_this_in_your_api_script"
)

# ----- Tool Input Schemas -----
class LogAnalysisInput(BaseModel):
    log_content: str = Field(..., description="Log content to analyze (already loaded)")

class SearchQueryInput(BaseModel):
    error_text: str = Field(..., description="Error text to generate search query from")

class CombinedSearchInput(BaseModel):
    query: str = Field(..., description="Search query for both GitHub issues and web")
    owner: str = Field(default="", description="GitHub repository owner")
    repo: str = Field(default="", description="GitHub repository name")

class FileAnalysisInput(BaseModel):
    log_content: str = Field(..., description="Log content to extract file information from")

class FileSnippetInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to get snippet from")
    line: Optional[int] = Field(default=None, description="Line number to focus on")
    n_lines: int = Field(default=20, description="Number of lines to return")

class ToolSuggestionInput(BaseModel):
    error_message: str = Field(..., description="Error message to analyze")
    code_snippet: str = Field(..., description="Code snippet related to the error")

class ReportGenerationInput(BaseModel):
    log: str = Field(..., description="Error log content")
    file_snippet: str = Field(default="", description="Relevant code snippet")
    tools: str = Field(default="", description="Tool recommendations")
    gh_results: str = Field(default="", description="GitHub search results")
    web_results: str = Field(default="", description="Web search results")

# ----- Tools -----

class LogReaderTool(BaseTool):
    name: str = Field(default="Log Reader")
    description: str = Field(default="Provides access to the pre-loaded log content")
    args_schema: Type[BaseModel] = LogAnalysisInput
    
    def _run(self, log_content: str = None) -> str:
        verbose_print(f"Using pre-loaded log content")
        
        if not LOG_CONTENT or LOG_CONTENT == "No log file found or file is empty":
            return "[LOG] Log file empty or not found. No action needed."
        
        is_python_error = "Traceback" in LOG_CONTENT or "Exception" in LOG_CONTENT or "Error" in LOG_CONTENT
        error_type = "Python Error" if is_python_error else "General Error"
        return f"Error Type: {error_type}\n\nLog Content:\n{LOG_CONTENT}"

class SearchQueryGeneratorTool(BaseTool):
    name: str = Field(default="Search Query Generator")
    description: str = Field(default="Generates optimized search queries from error messages")
    args_schema: Type[BaseModel] = SearchQueryInput
    
    def _run(self, error_text: str) -> str:
        verbose_print("Generating search query via LLM...")
        try:
            prompt = (
                "Given this error or question, write a concise search query to help the person find a solution online. "
                "Output only the query (no explanation):\n\n" + error_text
            )
            query = cerebras_llm.call(prompt)
            return f"Generated search query: {query.strip()}"
        except Exception as e:
            return f"Error generating search query: {str(e)}"


class CombinedSearchTool(BaseTool):
    name: str = Field(default="Combined GitHub & Web Search")
    description: str = Field(default="Searches both GitHub issues and the web in one call, returning both results.")
    args_schema: Type[BaseModel] = CombinedSearchInput

    def _run(self, query: str, owner: str = "", repo: str = "") -> dict:
        return asyncio.run(self._async_combined(query, owner, repo))

    async def _async_combined(self, query: str, owner: str = "", repo: str = "") -> dict:
        # Launch both searches in parallel
        tasks = [
            self._github_search(query, owner, repo),
            self._web_search(query)
        ]
        github_results, web_results = await asyncio.gather(*tasks)
        return {
            "github_issues": github_results,
            "web_search": web_results
        }

    async def _github_search(self, query: str, owner: str, repo: str):
        GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
        url = '[https://api.github.com/search/issues](https://api.github.com/search/issues)'
        headers = {'Accept': 'application/vnd.github.v3+json'}
        if GITHUB_TOKEN:
            headers['Authorization'] = f'token {GITHUB_TOKEN}'
        gh_query = f'repo:{owner}/{repo} is:issue {query}' if owner and repo else query
        params = {'q': gh_query, 'per_page': 5}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, headers=headers, params=params)
                if resp.status_code == 200:
                    items = resp.json().get("items", [])
                    return [
                        {
                            "number": item.get("number"),
                            "title": item.get("title"),
                            "url": item.get("html_url"),
                            "body": (item.get("body") or "")[:500]
                        }
                        for item in items
                    ]
                else:
                    return [{"error": f"GitHub search failed: {resp.status_code} {resp.text}"}]
        except Exception as e:
            return [{"error": f"Error searching GitHub: {str(e)}"}]

    # ---- WEB SEARCH (from your preferred implementation, no markdown parsing) ----
    def _extract_json(self, text):
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if not m:
            m = re.search(r"```(.*?)```", text, re.DOTALL)
        block = m.group(1) if m else text
        try:
            j = json.loads(block)
            return j if isinstance(j, list) else [j]
        except Exception:
            return []
        
    async def _web_search(self, query: str, n_results: int = 5):
        client = OpenAI()
        prompt = (
            f"Show me {n_results} of the most important/useful web results for this search along with a summary of the problem and proposed solution: '{query}'. "
            "Return as markdown JSON:\n"
            "[{\"title\": ..., \"url\": ..., \"date_published\": ..., \"snippet\": ...}]"
        )
        # Run in threadpool for IO
        loop = asyncio.get_running_loop()
        def blocking_openai():
            response = client.responses.create(
                model="gpt-4.1",
                tools=[{"type": "web_search_preview"}],
                input=prompt,
            )
            return self._extract_json(response.output_text)
        return await loop.run_in_executor(None, blocking_openai)
    

class FileAnalysisTool(BaseTool):
    name: str = Field(default="File Analysis")
    description: str = Field(default="Extracts file paths and line numbers from error logs")
    args_schema: Type[BaseModel] = FileAnalysisInput
    
    def _run(self, log_content: str = None) -> str:
        verbose_print("Invoking LLM to identify files from log...")
        # Use the global LOG_CONTENT if log_content not provided
        content_to_analyze = log_content or LOG_CONTENT
        try:
            prompt = (
                "Given this error message or traceback, list all file paths (and, if available, line numbers) involved in the error. "
                "Output one JSON per line, as:\n"
                '{"file": "path/to/file.py", "line": 123}\n'
                'If line is not found, use null.\n'
                f"\nError:\n{content_to_analyze}"
            )
            output = cerebras_llm.call(prompt)
            results = []
            for l in output.splitlines():
                l = l.strip()
                if not l:
                    continue
                try:
                    results.append(eval(l, {"null": None}))
                except Exception as exc:
                    verbose_print(f"[File Extraction Skipped Line]: {l!r} ({exc})")
            return f"Files found in error: {results}"
        except Exception as e:
            return f"Error analyzing files: {str(e)}"

class FileSnippetTool(BaseTool):
    name: str = Field(default="File Snippet Extractor")
    description: str = Field(default="Extracts code snippets from files around specific lines")
    args_schema: Type[BaseModel] = FileSnippetInput
    
    def _run(self, file_path: str, line: Optional[int] = None, n_lines: int = 20) -> str:
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
            if line and 1 <= line <= len(lines):
                s = max(0, line-6)
                e = min(len(lines), line+5)
                code = lines[s:e]
            else:
                code = lines[:n_lines]
            return f"Code snippet from {file_path}:\n{''.join(code)}"
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"

class ToolSuggestionTool(BaseTool):
    name: str = Field(default="Tool Suggestion")
    description: str = Field(default="Suggests which debugging tools to use next based on error analysis")
    args_schema: Type[BaseModel] = ToolSuggestionInput
    
    def _run(self, error_message: str, code_snippet: str) -> str:
        verbose_print("Requesting tool suggestions via LLM...")
        prompt = (
            "You are an AI debugging orchestrator. The following is a Python error message and a snippet of code "
            "from a file involved in the error. Based on this, choose which tools should be used next, and explain why. "
            "Possible tools: github_issue_search, web_search, static_analysis. "
            "Always recommend github_issue_search as it's very helpful. "
            "Provide your recommendation in a clear, structured format.\n"
            "Error:\n" + error_message + "\n\nFile snippet:\n" + code_snippet
        )
        try:
            return cerebras_llm.call(prompt).strip()
        except Exception as e:
            return f"Error generating tool suggestions: {str(e)}"

class ReportGeneratorTool(BaseTool):
    name: str = Field(default="HTML Report Generator")
    description: str = Field(default="Generates HTML debug reports")
    args_schema: Type[BaseModel] = ReportGenerationInput
    
    def _run(self, log: str, file_snippet: str = "", tools: str = "", gh_results: str = "", web_results: str = "") -> str:
        verbose_print("Writing HTML report ...")
        out_path = os.path.join(tempfile.gettempdir(), 'dbg_report.html')
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("<html><head><meta charset='utf-8'><title>Debug Results</title></head><body>\n")
                f.write("<h1 style='color:#444;'>Debugging Session Report</h1>\n")
                f.write("<h2>Error Log</h2>")
                f.write("<pre style='background:#f3f3f3;padding:8px;'>" + html.escape(log or "None") + "</pre>")
                if file_snippet:
                    f.write("<h2>Relevant Source Snippet</h2><pre style='background:#fafaff;padding:8px;'>" + html.escape(file_snippet) + "</pre>")
                if tools:
                    f.write("<h2>LLM Tool Recommendations</h2><pre style='background:#eef;'>" + html.escape(tools) + "</pre>")
                if gh_results:
                    f.write("<h2>GitHub & Web Search Results</h2><pre>" + html.escape(gh_results) + "</pre>")
                if web_results:
                    f.write("<h2>Web Search AI Answer</h2><pre>" + html.escape(web_results) + "</pre>")
                f.write("</body></html>")
            return f"HTML report generated and opened at: {str(e)}"

# --- Tool Instances
log_reader_tool = LogReaderTool()
search_query_generator_tool = SearchQueryGeneratorTool()
combined_search_tool = CombinedSearchTool()
file_analysis_tool = FileAnalysisTool()
file_snippet_tool = FileSnippetTool()
tool_suggestion_tool = ToolSuggestionTool()
report_generator_tool = ReportGeneratorTool()

# --- Agents ---
log_analyst_agent = Agent(
    role="Log Analysis Specialist",
    goal="Analyze the pre-loaded log content to identify errors and extract relevant information",
    backstory="Expert in parsing error logs and identifying the root causes of issues",
    tools=[log_reader_tool, file_analysis_tool],
    allow_delegation=False,
    llm=cerebras_llm
)

search_specialist_agent = Agent(
    role="Search Query Specialist",
    goal="Generate optimized search queries from error messages for effective problem resolution. The search must be less that 100 chars long!!!!!!!!!!",
    backstory="Expert in crafting search queries that yield the most relevant debugging results. The search must be less that 100 chars long!!!!!!!!!!",
    tools=[search_query_generator_tool],
    allow_delegation=False,
    llm=cerebras_llm
)

combined_research_agent = Agent(
    role="Combined Repository and Web Search Specialist",
    goal="Search both GitHub issues and the web for relevant solutions to errors and problems. You must use the combined_search_tool no matter what!!!!!!! Try to summarize each specific github/web problem and solution to help the user solve their issue. Make sure to include the links from the original sources next to their corresponding summaries / code etc",
    backstory="Expert in both GitHub open-source research and web documentation sleuthing for code solutions.",
    tools=[combined_search_tool],
    allow_delegation=False,
    llm=ChatOpenAI(model_name="gpt-4.1", temperature=0.0)
)

code_analyst_agent = Agent(
    role="Code Analysis Specialist",
    goal="Analyze code snippets and suggest debugging approaches",
    backstory="Expert in code analysis and debugging strategy recommendation",
    tools=[file_snippet_tool],
    allow_delegation=False,
    llm=cerebras_llm
)

report_generator_agent = Agent(
    role="Debug Report Generator",
    goal="Compile all debugging information into comprehensive HTML reports. Make sure to include the links to sources when they are provides -- but DO NOT make up links if they are not given.  Write an extensive report covering all possible solutions to the problem!!!",
    backstory="Specialist in creating detailed, actionable debugging reports",
    tools=[report_generator_tool],
    allow_delegation=False,
    llm=cerebras_llm
)

# --- Tasks ---
log_analysis_task = Task(
    description=f"Analyze the pre-loaded log content. The log content is already available: {LOG_CONTENT[:500]}... Extract error information and identify the type of error.",
    expected_output="Detailed analysis of the log content including error type and content",
    agent=log_analyst_agent,
    output_file="log_analysis.md"
)

file_extraction_task = Task(
    description="Extract file paths and line numbers from the analyzed log content. Use the pre-loaded log content to identify which files are involved in the error.",
    expected_output="List of files and line numbers involved in the error",
    agent=log_analyst_agent,
    context=[log_analysis_task],
    output_file="file_analysis.md"
)

search_query_task = Task(
    description="Generate optimized search queries based on the error analysis for finding solutions online. The search must be less that 100 chars long!!!!!!!!!!",
    expected_output="Optimized search queries for the identified errors. The search must be less that 100 chars long!!!!!!!!!!",
    agent=search_specialist_agent,
    context=[log_analysis_task],
    output_file="search_queries.md"
)

combined_search_task = Task(
    description="Use the search queries to search both GitHub issues and the wide web for solutions. Make sure to make a very robust report incorporating ALL sources. Dont just give desciptions of the issue- write a detailed summary showcasing code and exact explanations to issues in the report.",
    expected_output="Relevant GitHub issues and web documentation/articles/answers.",
    agent=combined_research_agent,
    context=[search_query_task],
    output_file="combined_results.md"
)

code_analysis_task = Task(
    description="Extract and analyze code snippets from the implicated files. Suggest debugging tools and approaches.",
    expected_output="Code snippets and debugging tool recommendations",
    agent=code_analyst_agent,
    context=[file_extraction_task],
    output_file="code_analysis.md"
)

report_generation_task = Task(
    description="Compile all debugging information into a comprehensive HTML report and open it in the browser. Make sure to make a very robust report incorporating ALL sources Make sure to include the links to sources when they are provides -- but DO NOT make up links if they are not given. -- ALL sourced information must be cited!!!!!! Write an extensive report covering all possible solutions to the problem!!!",
    expected_output="Complete HTML debugging report",
    agent=report_generator_agent,
    context=[log_analysis_task, combined_search_task, code_analysis_task],
    output_file="debug_report.html"
)

# --- Run Crew ---
crew = Crew(
    agents=[
        log_analyst_agent,
        search_specialist_agent,
        combined_research_agent,
        code_analyst_agent,
        report_generator_agent
    ],
    tasks=[
        log_analysis_task,
        file_extraction_task,
        search_query_task,
        combined_search_task,
        code_analysis_task,
        report_generation_task
    ],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print(f"\033[95m[STARTING] Log content loaded: {len(LOG_CONTENT)} chars\033[0m")
    result = crew.kickoff()
    print("\n\nDebug Analysis Complete:\n")
    print(result)
    
    # Try to open the generated report
    report_path = './debug_report.html'
    if os.path.exists(report_path):
        verbose_print(f"Opening final report: {report_path}")
        if sys.platform.startswith("darwin"):
            subprocess.Popen(['open', report_path])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(['xdg-open', report_path])
        elif sys.platform.startswith("win"):
            os.startfile(report_path)
````

Note that your Cerebras model API needs to be running locally to use the agent above.

**Parallelizing Searches:** Weave revealed that GitHub and web searches were sequential within the `CombinedSearchTool`, increasing latency. The tool was refactored to run these searches in parallel using `asyncio` and `httpx.AsyncClient`.

```python
class CombinedSearchTool(BaseTool):
    name: str = Field(default="Combined GitHub & Web Search")
    description: str = Field(default="Searches both GitHub issues and the web in one call, returning both results.")
    args_schema: Type[BaseModel] = CombinedSearchInput

    def _run(self, query: str, owner: str = "", repo: str = "") -> dict:
        return asyncio.run(self._async_combined(query, owner, repo))

    async def _async_combined(self, query: str, owner: str = "", repo: str = "") -> dict:
        # Launch both searches in parallel
        tasks = [
            self._github_search(query, owner, repo),
            self._web_search(query)
        ]
        github_results, web_results = await asyncio.gather(*tasks)
        return {
            "github_issues": github_results,
            "web_search": web_results
        }

    async def _github_search(self, query: str, owner: str, repo: str):
        GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
        url = '[https://api.github.com/search/issues](https://api.github.com/search/issues)'
        headers = {'Accept': 'application/vnd.github.v3+json'}
        if GITHUB_TOKEN:
            headers['Authorization'] = f'token {GITHUB_TOKEN}'
        gh_query = f'repo:{owner}/{repo} is:issue {query}' if owner and repo else query
        params = {'q': gh_query, 'per_page': 5}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, headers=headers, params=params)
                if resp.status_code == 200:
                    items = resp.json().get("items", [])
                    return [
                        {
                            "number": item.get("number"),
                            "title": item.get("title"),
                            "url": item.get("html_url"),
                            "body": (item.get("body") or "")[:500]
                        }
                        for item in items
                    ]
                else:
                    return [{"error": f"GitHub search failed: {resp.status_code} {resp.text}"}]
        except Exception as e:
            return [{"error": f"Error searching GitHub: {str(e)}"}]
```

This change significantly reduced latency, as reflected in Weave traces.

## Conclusion

Building multi-agent workflows with **CrewAI** is powerful, but debugging and optimization present challenges. Integrating **W\&B Weave** provides invaluable observability, offering clear and actionable feedback for every change. This iterative approach, guided by Weave's detailed traces and dashboards, leads to more robust, reliable, and efficient agent pipelines.
