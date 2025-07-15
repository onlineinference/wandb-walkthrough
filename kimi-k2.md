
# Advanced Kimi K2 Observability with W\&B Weave

This tutorial demonstrates how to leverage W&B Weave's powerful observability features to monitor, debug, and optimize Kimi K2's performance across various coding and agentic use cases. We'll specifically use OpenRouter as the API endpoint for Kimi K2, providing a convenient and free-tier-friendly way to access this powerful model.

-----

## Prerequisites

Before you begin, ensure you have the necessary libraries installed and your W\&B and OpenRouter accounts set up.

```bash
pip install wandb weave openai
```

You'll need:

  * A **Weights & Biases account and API key** (available from wandb.ai/settings).
  * An **OpenRouter account and API key** (available from openrouter.ai/keys).

-----

## Step 1: Initialize Weave and Configure Kimi K2 via OpenRouter

First, you'll set up your Python environment by initializing W&B Weave and configuring the OpenAI client to point to OpenRouter's Kimi K2 endpoint.

```python
import weave
import openai
from typing import Dict, List, Any
import time
import json

# Initialize Weave with a clear project name.
# This creates a dedicated space in your W&B dashboard for Kimi K2 logs.
# You will be prompted to log in to W&B if you haven't already.
weave.init(
    project_name="kimi-k2-deep-observability",
    # entity="your-wandb-entity" # Optional: specify your W&B team/entity if working in a team
)

# Configure the OpenAI client to use OpenRouter's API for Kimi K2.
# OpenRouter's API is largely compatible with OpenAI's.
# IMPORTANT: Replace "YOUR_OPENROUTER_API_KEY" with your actual OpenRouter API key.
# It should start with 'sk-'.
client = openai.OpenAI(
    api_key="YOUR_OPENROUTER_API_KEY",
    base_url="https://api.openrouter.ai/api/v1"
)

# Define the model name for Kimi K2 on OpenRouter.
# Using 'moonshotai/kimi-k2:free' targets the free tier of the model.
# Be aware of OpenRouter's free tier rate limits (e.g., 50 requests/day, 20/min).
KIMI_K2_MODEL_NAME = "moonshotai/kimi-k2:free"
```

-----

## Step 2: Create Instrumented Functions with Rich Metadata

Weave allows you to wrap your functions with `@weave.op()` decorators to automatically track their execution, inputs, and outputs. This is where you'll define your interactions with Kimi K2, embedding valuable metadata for detailed analysis in the Weights & Biases dashboard.

````python
@weave.op()
def generate_code_with_context(
    prompt: str,
    language: str = "python",
    complexity: str = "medium",
    include_tests: bool = False
) -> Dict[str, Any]:
    """
    Calls Kimi K2 to generate code based on a prompt,
    tracking various parameters and returning structured output for observability.
    """

    # Build an enhanced prompt that provides Kimi K2 with more context.
    # This helps guide the model's generation and ensures better quality outputs.
    enhanced_prompt = f"""
    Language: {language}
    Complexity: {complexity}
    Include tests: {include_tests}
    
    Task: {prompt}
    
    Please provide clean, well-documented code.
    """

    start_time = time.time()

    # Call Kimi K2 via the OpenRouter-configured OpenAI client
    response = client.chat.completions.create(
        model=KIMI_K2_MODEL_NAME, # Use the defined Kimi K2 model name (including ':free' suffix)
        messages=[
            {"role": "system", "content": "You are an expert programmer. Write clean, efficient code with clear comments."},
            {"role": "user", "content": enhanced_prompt}
        ],
        temperature=0.6, # Recommended temperature for balanced creativity and accuracy with Kimi K2
        max_tokens=1000,
        # Optional: Add OpenRouter specific headers for tracking/ranking
        extra_headers={
            "HTTP-Referer": "https://kimi-k2-tutorial.example.com", # Your site URL for rankings on openrouter.ai
            "X-Title": "Kimi K2 Weave Tutorial", # Your app title for rankings on openrouter.ai
        }
    )

    end_time = time.time()

    # Weave automatically captures basic OpenAI client metrics (tokens, cost).
    # By returning a dictionary, we can add custom metadata that will appear in the trace.
    return {
        "generated_code": response.choices[0].message.content,
        "language": language,
        "complexity": complexity,
        "include_tests": include_tests, # Log input parameter for easier filtering later
        "response_time_seconds": end_time - start_time,
        "token_usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        },
        "model_info": {
            "model_id": response.model, # The actual model ID used by OpenRouter/Kimi K2
            "finish_reason": response.choices[0].finish_reason
        }
    }

@weave.op()
def evaluate_code_quality(code: str, language: str) -> Dict[str, Any]:
    """
    Uses Kimi K2 to evaluate the quality of a given code snippet.
    This creates a feedback loop, demonstrating Kimi K2's ability to "self-critique"
    or provide an objective assessment.
    """

    evaluation_prompt = f"""
    Please evaluate this {language} code on a scale of 1-10 for the following criteria:
    - Correctness (Is the code logically sound and bug-free?)
    - Readability (Is the code easy to understand and well-documented?)
    - Efficiency (Is the code optimized for performance?)
    - Adherence to Best Practices (Does it follow common {language} conventions and patterns?)
    
    Code to evaluate:
    ```{language}
    {code}
    ```
    
    Provide your scores and brief explanations for each criterion in JSON format.
    Example expected output:
    {{
      "correctness": {{ "score": 9, "explanation": "..." }},
      "readability": {{ "score": 8, "explanation": "..." }},
      "efficiency": {{ "score": 7, "explanation": "..." }},
      "best_practices": {{ "score": 8, "explanation": "..." }}
    }}
    """

    response = client.chat.completions.create(
        model=KIMI_K2_MODEL_NAME,
        messages=[{"role": "user", "content": evaluation_prompt}],
        temperature=0.3, # Lower temperature for more consistent and factual evaluation
        max_tokens=500,
        extra_headers={
            "HTTP-Referer": "https://kimi-k2-tutorial.example.com",
            "X-Title": "Kimi K2 Weave Tutorial (Evaluator)",
        }
    )

    try:
        # Attempt to parse the evaluation response as JSON.
        # Weave will log parsing errors if they occur, aiding debugging.
        evaluation_content = response.choices[0].message.content
        evaluation = json.loads(evaluation_content)
        return {
            "evaluation_scores": evaluation,
            "evaluator_model_id": response.model,
            "evaluation_tokens": response.usage.total_tokens
        }
    except json.JSONDecodeError as e:
        # If Kimi K2 doesn't return perfect JSON, Weave still logs the raw response,
        # which is invaluable for debugging prompt engineering.
        return {
            "evaluation_scores": {"error": f"JSON parsing failed: {e}"},
            "raw_evaluation_response": response.choices[0].message.content,
            "evaluator_model_id": response.model,
            "evaluation_tokens": response.usage.total_tokens
        }

@weave.op()
def multi_step_coding_agent(task_description: str) -> Dict[str, Any]:
    """
    Demonstrates Kimi K2's agentic capabilities by orchestrating a multi-step workflow.
    Each sub-step (planning, code generation, evaluation) is tracked separately
    within the overall trace, providing deep insight into the agent's decision-making.
    """

    # Step 1: Kimi K2 acts as a planner to break down the complex task.
    # This simulates a "thought" process, which is a hallmark of agentic AI.
    print(f"  --> Agent Planning for: {task_description}")
    planning_response = client.chat.completions.create(
        model=KIMI_K2_MODEL_NAME,
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant that can break down complex programming tasks into actionable steps."
        }, {
            "role": "user",
            "content": f"Break down this coding task into logical, sequential steps for a developer to follow: {task_description}"
        }],
        temperature=0.5,
        max_tokens=300,
        extra_headers={
            "HTTP-Referer": "https://kimi-k2-tutorial.example.com",
            "X-Title": "Kimi K2 Weave Tutorial (Planner)",
        }
    )

    task_plan = planning_response.choices[0].message.content
    print(f"  --> Plan generated:\n{task_plan}\n")

    # Step 2: Generate the code based on the original task.
    # This uses our previously defined, instrumented function.
    print("  --> Agent generating code...")
    code_result = generate_code_with_context(
        prompt=task_description,
        language="python",
        complexity="medium",
        include_tests=True
    )
    print("  --> Code generated.")

    # Step 3: Evaluate the generated code.
    # This showcases a self-correction or verification step within the agent.
    print("  --> Agent evaluating generated code...")
    evaluation = evaluate_code_quality(
        code=code_result["generated_code"],
        language="python"
    )
    print("  --> Code evaluation complete.")

    return {
        "task_description": task_description,
        "agent_plan": task_plan,
        "generated_code_result": code_result,
        "code_evaluation_result": evaluation,
        "workflow_completed": True
    }
````

-----

## Step 3: Execute and Monitor Multiple Scenarios

Now you'll run your instrumented functions with different inputs. Each call will generate a trace that Weave captures and sends to your Weights & Biases dashboard, providing a rich dataset for analysis.

````python
# Test different types of coding tasks to observe Kimi K2's performance
test_scenarios = [
    {
        "name": "Basic Algorithm",
        "prompt": "Write a Python function to check if a string is a palindrome, ignoring case and non-alphanumeric characters.",
        "language": "python",
        "complexity": "easy"
    },
    {
        "name": "Data Manipulation",
        "prompt": "Write a Python function to flatten a nested list of integers. Example: [[1,2],[3,[4,5]],6] -> [1,2,3,4,5,6]",
        "language": "python",
        "complexity": "medium"
    },
    {
        "name": "Simple Web Utility",
        "prompt": "Create a simple Python Flask API endpoint that returns the current server time.",
        "language": "python",
        "complexity": "medium"
    }
]

# Execute scenarios and collect results
print("--- Running Individual Code Generation Scenarios ---")
results = []
for scenario in test_scenarios:
    print(f"\n🔄 Running scenario: {scenario['name']}")

    # Generate code with enhanced tracking
    result = generate_code_with_context(
        prompt=scenario["prompt"],
        language=scenario["language"],
        complexity=scenario["complexity"],
        include_tests=True # Always include tests for these examples
    )

    # Add scenario metadata to the result for easier filtering and comparison in Weave
    result["scenario_name"] = scenario["name"]
    result["scenario_complexity"] = scenario["complexity"]

    results.append(result)

    print(f"✅ Scenario completed in {result['response_time_seconds']:.2f}s")
    print(f"📊 Tokens used: {result['token_usage']['total_tokens']}")
    print(f"Generated code snippet:\n```python\n{result['generated_code'][:200]}...\n```") # Print a snippet

# Demonstrate a more complex, multi-step agentic workflow
print("\n--- Running Multi-Step Agentic Workflow ---")
workflow_result = multi_step_coding_agent(
    "Develop a Python script that uses requests to fetch data from a public API (e.g., JSONPlaceholder /posts), processes it to count words in titles, and then writes the top 5 most common words and their counts to a CSV file."
)

print("\n✅ Multi-step workflow completed. Check your W&B Weave dashboard for details!")
````

-----

## Step 4: Analyzing Results in W\&B Weave

After running this script, you'll see a link in your console (look for the 🟢 or 🍩 emoji) directing you to your dashboard. Click this link to explore the rich observability data.

### Key Features to Explore in the W\&B Weave UI:

  * **Trace Visualization:** Navigate to the "Traces" tab. You'll see individual runs for `generate_code_with_context` and a nested, multi-step trace for `multi_step_coding_agent`.
  * **Click on a trace:** See the complete execution flow, including inputs (prompts, parameters), outputs (generated code, evaluation JSON), and intermediate steps.
    For `multi_step_coding_agent`, you'll observe nested calls to `generate_code_with_context` and `evaluate_code_quality`, providing a clear visualization of how Kimi K2 planned and executed the task. This nested view is incredibly powerful for understanding complex agentic behaviors.
  * **Performance Metrics:** In the trace view, observe the `response_time_seconds` and `token_usage` for each operation. These metrics help you identify bottlenecks, understand the computational cost of different Kimi K2 interactions, and track efficiency over time.
  * **Input/Output Inspection:** For every logged operation, you can inspect the exact prompt sent to Kimi K2 and the complete response received. This is crucial for debugging Kimi K2's outputs and refining your prompt engineering strategies.
  * **Custom Metadata:** Notice how the `language`, `complexity`, `include_tests`, `scenario_name`, and `scenario_complexity` fields appear in your traces. These custom metadata fields make it easy to filter, group, and analyze runs in the Weights & Biases UI for targeted comparisons.
  * **Comparison Views:** W\&B Weave allows you to select multiple runs and compare them side-by-side. You can easily see how Kimi K2's output or performance varies with different temperature settings, max\_tokens, or prompt variations. This A/B testing capability is invaluable for optimization.
  * **Error Analysis:** If `evaluate_code_quality` encountered a `JSONDecodeError` (e.g., Kimi K2 failed to produce perfect JSON), you'll see it logged in the trace along with the `raw_evaluation_response`. This helps you quickly diagnose why Kimi K2 might not be adhering to a specific output format.

After visiting your Weave dashboard you'll find a list of your traces, including those that failed, assisting in troubleshooting errors you might not know existed:

You can further dig into individual traces to explore the inputs and outputs, to explore how things are working, where they can be improved and provide not just explainability but observability.

![Viewing traces in Weave](https://github.com/onlineinference/wandb-walkthrough/blob/main/images/f00c2222.png)

-----

## Step 5: (Optional) Custom Evaluation and Feedback Loop

While Weave automatically logs a lot, you can further enhance your observability by explicitly logging custom aggregate metrics or evaluation results. This is useful for building dashboards that track long-term performance or specific KPIs.

```python
@weave.op()
def batch_code_generation_with_metrics(prompts: List[str]) -> Dict[str, Any]:
    """
    Processes multiple prompts in a batch and collects comprehensive aggregate metrics.
    """

    batch_results = []
    total_tokens_sum = 0
    total_time_sum = 0
    successful_requests_count = 0

    for i, prompt in enumerate(prompts):
        print(f"  Processing batch item {i+1}/{len(prompts)}")
        try:
            result = generate_code_with_context(
                prompt=prompt,
                language="python",
                complexity="medium"
            )

            batch_results.append({
                "prompt_index": i,
                "prompt": prompt,
                "success": True,
                "result": result
            })

            total_tokens_sum += result["token_usage"]["total_tokens"]
            total_time_sum += result["response_time_seconds"]
            successful_requests_count += 1

        except Exception as e:
            # Weave will log the exception details automatically
            print(f"  Error processing prompt {i}: {e}")
            batch_results.append({
                "prompt_index": i,
                "prompt": prompt,
                "success": False,
                "error": str(e)
            })

    # Calculate aggregate metrics for the batch
    avg_tokens_per_request = total_tokens_sum / len(prompts) if prompts else 0
    avg_response_time = total_time_sum / successful_requests_count if successful_requests_count > 0 else 0
    success_rate = successful_requests_count / len(prompts) if prompts else 0

    return {
        "batch_processing_summary": {
            "total_requests": len(prompts),
            "successful_requests": successful_requests_count,
            "success_rate": success_rate,
            "total_tokens_generated": total_tokens_sum,
            "avg_tokens_per_request": avg_tokens_per_request,
            "avg_response_time_seconds": avg_response_time,
            "total_processing_time_seconds": total_time_sum
        },
        "individual_batch_results": batch_results # Can be used to drill down into individual failures
    }

# Run batch processing
batch_prompts = [
    "Write a Python function to reverse a linked list.",
    "Create a class for a simple calculator with add, subtract, multiply, and divide methods.",
    "Implement the bubble sort algorithm in Python.",
    "Write a Python function to validate if a string is a valid email address.",
    "Create a Python decorator for timing function execution and printing the duration."
]

print("\n--- Running Batch Processing with Aggregate Metrics ---")
batch_summary_result = batch_code_generation_with_metrics(batch_prompts)

# Display summary metrics in the console and they'll also be logged by Weave
metrics = batch_summary_result["batch_processing_summary"]
print(f"""
📈 Batch Processing Summary:
- Total Requests: {metrics['total_requests']}
- Successful Requests: {metrics['successful_requests']}
- Success Rate: {metrics['success_rate']:.2%}
- Average Response Time: {metrics['avg_response_time_seconds']:.2f}s
- Average Tokens per Request: {metrics['avg_tokens_per_request']:.0f}
- Total Processing Time for Batch: {metrics['total_processing_time_seconds']:.2f}s
""")

# Example of how you might create a dataset of evaluations, if you were manually
# reviewing and rating generated code (human-in-the-loop) or had a more
# sophisticated automated testing setup.
@weave.op()
def create_human_evaluation_dataset(generated_codes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Placeholder for creating a dataset of human or automated code evaluations.
    This demonstrates how to log a 'dataset' artifact to W&B for further analysis.
    """
    
    # In a real scenario, you might have a process where human reviewers
    # score the generated_codes and then you log those scores.
    # For this example, we'll just prepare a structure.
    
    evaluation_entries = []
    for i, code_info in enumerate(generated_codes):
        # Simulate some evaluation here, or integrate with an external testing framework
        simulated_score = 7 + (i % 3) # Dummy score for demonstration
        evaluation_entries.append({
            "sample_id": i,
            "prompt": code_info.get("prompt", "N/A"),
            "generated_code": code_info["generated_code"],
            "simulated_human_score": simulated_score,
            "model_response_time": code_info["response_time_seconds"]
        })
    
    # Weave automatically logs the return value of @weave.op functions.
    # This dictionary will appear as a logged object in the trace.
    return {
        "evaluation_dataset_summary": {
            "dataset_size": len(evaluation_entries),
            "timestamp": time.time(),
            "average_simulated_score": sum([e["simulated_human_score"] for e in evaluation_entries]) / len(evaluation_entries) if evaluation_entries else 0
        },
        "evaluation_details": evaluation_entries
    }

# Create evaluation dataset from our generated code (using dummy scores for demonstration)
print("\n--- Creating Simulated Evaluation Dataset ---")
evaluation_dataset_output = create_human_evaluation_dataset([r for r in results if r.get("generated_code")])
print(f"📋 Created a simulated evaluation dataset with {evaluation_dataset_output['evaluation_dataset_summary']['dataset_size']} samples.")
print(f"Average simulated score: {evaluation_dataset_output['evaluation_dataset_summary']['average_simulated_score']:.2f}")

print("\nTutorial execution complete. Check your W&B Weave dashboard for detailed observability!")
```
![Reviewing batch processes in Weave](https://github.com/onlineinference/wandb-walkthrough/blob/main/images/c579d531.png)
