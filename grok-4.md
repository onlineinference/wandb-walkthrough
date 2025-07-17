# Code Generation and Observability with Grok 4 and W&B Weave

This tutorial will guide you through using xAI's cutting-edge Grok 4 model for advanced code generation and debugging, while simultaneously leveraging W&B Weave for comprehensive observability. By integrating Weave, you'll gain a powerful way to track every prompt, Grok 4’s response, and your evaluation results, creating a robust and transparent AI-powered development workflow.

-----

## 1. Setting Up Your Environment: Grok 4, OpenRouter, and W&B Weave

Before we dive into code generation, let's get our environment ready by setting up access to Grok 4 via OpenRouter and initializing W&B Weave for logging.

### 1.1 Create OpenRouter and W&B Accounts & API Keys:

  * **OpenRouter:** Sign up on the OpenRouter website (openrouter.ai). Once logged in, navigate to the API keys section and create a new API key. This key will authenticate your requests to the OpenRouter API. Keep it secure.
  * **Weights & Biases:** Create a free Weights & Biases account. After signing up, navigate to [https://wandb.ai/authorize](https://wandb.ai/authorize) to find your Weights & Biases API key. This key is essential for logging data to your W&B dashboard.

### 1.2 Enable Access to xAI’s Grok Model on OpenRouter:

On the OpenRouter platform, locate xAI’s Grok 4 in the model list (labeled as `x-ai/grok-4`). You might need to agree to specific terms of service or ensure you have a payment method on file, as Grok 4 is a premium model.

### 1.3 Install Necessary Python Libraries:

Ensure you have the `requests` library (for OpenRouter API calls) and `weave` (for W&B observability) installed. Open your terminal or command prompt and run:

```bash
pip install requests weave
```

### 1.4 Configure API Access and Initialize Weave in Your Script:

Now, let's set up the Python script where all our interactions will happen. We'll define two key functions wrapped with `weave.op()`: `call_grok4` to interact with the model and `evaluate_prime_function` to test its output. Wrapping these with `@weave.op()` automatically logs their inputs, outputs, and execution details to your W&B dashboard, creating detailed 'traces'.

**Important:** Replace `"YOUR_OPENROUTER_API_KEY"` with your actual key. It's recommended to use environment variables for API keys in real projects.

```python
import weave
import requests
import math
import os
import re

# --- W&B Weave Initialization ---
project_name = "gh-Grok4-CodeGen-Tutorial"
weave.init(project_name)

# --- OpenRouter API Configuration ---
# Secure way to input API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY") # Replace with your key if not using env var
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROK4_MODEL = "x-ai/grok-4"

# Define a Weave operation for calling Grok 4
@weave.op()
def call_grok4(prompt: str, context_messages: list = None) -> str:
    """Sends a prompt to Grok 4 via OpenRouter and returns the response."""
    if context_messages is None:
        context_messages = []
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:8888",  # Required for OpenRouter
        "X-Title": "Grok4-CodeGen-Tutorial"  # Optional but recommended
    }
    
    messages = context_messages + [{"role": "user", "content": prompt}]
    data = {
        "model": GROK4_MODEL,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(OPENROUTER_URL, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        # Check if response has the expected structure
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return f"Unexpected response format: {result}"
            
    except requests.exceptions.RequestException as e:
        return f"API request failed: {str(e)}"
    except KeyError as e:
        return f"Unexpected response structure: {str(e)}"

# Define a Weave operation to evaluate the generated code
@weave.op()
def evaluate_prime_function(code: str, test_cases: list) -> dict:
    """
    Executes the provided code and tests the is_prime function against test cases.
    Returns a dictionary of results for Weave to log.
    """
    results = {}
    total_tests = len(test_cases)
    passed_tests = 0

    try:
        # Create a clean namespace for code execution
        exec_globals = {"math": math}  # Provide math module
        
        # Execute the generated code to define the is_prime function
        exec(code, exec_globals)
        is_prime_func = exec_globals.get('is_prime')

        if not is_prime_func:
            raise ValueError("Generated code does not define 'is_prime' function.")

        # Test each case
        for num, expected in test_cases:
            try:
                actual = is_prime_func(num)
                is_correct = (actual == expected)
                if is_correct:
                    passed_tests += 1
                results[f"Test_{num}"] = {
                    "input": num,
                    "expected": expected,
                    "actual": actual,
                    "correct": is_correct
                }
            except Exception as test_error:
                results[f"Test_{num}"] = {
                    "input": num,
                    "expected": expected,
                    "actual": None,
                    "correct": False,
                    "error": str(test_error)
                }
                
    except Exception as e:
        results["execution_error"] = str(e)
        passed_tests = 0

    return {
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "accuracy": passed_tests / total_tests if total_tests > 0 else 0,
        "test_details": results
    }

print("Setup complete!")
```

You've now configured your environment and defined two `weave.op` functions. Every time `call_grok4` or `evaluate_prime_function` is called, Weave will automatically log their inputs, outputs, and execution details to your W&B dashboard, creating a 'trace' of the process.

-----

## 2\. Generate Initial Code with Grok 4

Let's begin by asking Grok 4 to write a Python function that checks if a number is prime. We'll use our `call_grok4` operation to send the prompt, ensuring this interaction is logged.

**User Prompt:** “Write a Python function `is_prime(n)` that returns `True` if `n` is a prime number and `False` otherwise. The function should be efficient for large `n` and include comments explaining the logic.”

```python
initial_prompt = """Write a Python function is_prime(n) that returns True if n is a prime number and False otherwise. The function should be efficient for large n and include comments explaining the logic. Only return the function, with no explanations."""

print("Sending initial prompt to Grok 4...")
# Calling our Weave-wrapped function logs this interaction
generated_code = call_grok4(initial_prompt) 
print("\n--- Grok 4 Generated Code ---")
print(generated_code)
print("-----------------------------\n")
```

After running this, navigate to your W&B project (e.g., `wandb.ai/your_wandb_username/Grok4-CodeGen-Tutorial`). You should see a new "trace" entry under the "Traces" tab, representing this `call_grok4` execution. Clicking on it will show you the exact prompt sent and the full code received, along with metadata like token usage and latency. It will look something like:

For our prime-checking function, Grok 4 generated:

```python
import math

def is_prime(n):
    """
    Checks if a number n is prime.
    
    A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.
    This function uses an efficient trial division method optimized for performance, checking divisibility
    up to the square root of n, with additional optimizations to skip multiples of 2 and 3.
    
    Time complexity: O(sqrt(n)), which is efficient for n up to around 10^18 on modern hardware.
    
    :param n: An integer to check for primality.
    :return: True if n is prime, False otherwise.
    """
    # Handle small numbers: primes start from 2, so anything <= 1 is not prime.
    if n <= 1:
        return False
    # 2 and 3 are primes.
    if n <= 3:
        return True
    # Eliminate multiples of 2 and 3 early, as they can't be prime (except 2 and 3 themselves).
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Now check for factors from 5 onwards, using the 6k ± 1 optimization.
    # All primes greater than 3 are of the form 6k ± 1.
    # We check i and i+2 (which is i+2 = (i+6)-4, but effectively covers the pattern) in steps of 6.
    # We only need to check up to sqrt(n) because if n has a factor larger than sqrt(n), 
    # it must also have a smaller one we've already checked.
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```

Grok 4 often outputs nicely formatted code blocks, including comments and handling edge cases. Always review the code for basic sanity: does it follow the prompt instructions? Are there obvious mistakes?

-----

## 3\. Test and Evaluate the Generated Code with Weave

Now, we'll verify if the generated code works correctly using a set of test cases. Our `evaluate_prime_function` operation will run the tests and automatically log the results to Weave, linking them to the code generation trace.

```python
test_cases = [
    (1, False),    # 1 is not prime
    (2, True),     # 2 is prime
    (3, True),     # 3 is prime
    (4, False),    # 4 is not prime
    (15, False),   # 15 is not prime
    (17, True),    # 17 is prime
    (97, True),    # 97 is prime
    (100, False),  # 100 is not prime
    (997, True)    # 997 is prime
]

print("Evaluating the generated code...")
# Calling our Weave-wrapped evaluation function logs the results
evaluation_results = evaluate_prime_function(generated_code, test_cases)
print("\n--- Evaluation Results ---")
print(evaluation_results)
print("--------------------------\n")
```

Go back to your Weights & Biases dashboard. The `call_grok4` trace you saw earlier should now be connected to an `evaluate_prime_function` trace. You can view the detailed `evaluation_results` dictionary, which includes `passed_tests`, `total_tests`, and `accuracy`, directly within the trace details. This immediately tells you how well Grok's initial output performed against your defined tests.

If you execute these tests (you'd typically copy the `is_prime` function generated by Grok into your Python environment or a notebook to be runnable for `evaluate_prime_function`), you should check the outputs against expected results. If a bug is found or the output isn't as expected, we move to the debugging step.

-----

## 4\. Debug and Refine with Grok 4 (Observed with Weave)

One of Grok 4’s strengths is its ability to help debug code it (or someone else) wrote. Suppose our testing revealed that `is_prime(1)` returned `True`, which is incorrect (even if the provided example code snippet is correct; we'll simulate this for demonstration purposes). We can feed this information back to Grok 4 to get a fix. Weave will log this next interaction as a new trace, allowing us to track the iterative refinement process.

````python
# Simulate a problematic code example for demonstration purposes
# In a real scenario, this would be the actual 'generated_code' from step 2 if it had a flaw.
problematic_code_example = """
def is_prime(n: int) -> bool:
    # Simulating a bug: This version might not handle n < 2 correctly
    if n % 2 == 0:
        return n == 2
    import math
    limit = int(math.sqrt(n)) + 1
    for divisor in range(3, limit, 2):
        if n % divisor == 0:
            return False
    return True
""" 

# Provide feedback to Grok 4 based on the evaluation findings
debugging_prompt = f"""The `is_prime` function you provided:
```python
{problematic_code_example}
````

Is incorrect for `n=1`. It returns `True` but should return `False`. Please fix the function to correctly handle `n=1` and ensure it remains efficient. Only return the corrected function, with no extra explanations."""

# Calling call\_grok4 again logs this new interaction

print("Sending debugging prompt to Grok 4...")
corrected\_code = call\_grok4(debugging\_prompt)
print("\\n--- Grok 4 Corrected Code ---")
print(corrected\_code)
print("------------------------------\\n")

# Re-evaluate the corrected code, logging the new evaluation with Weave

print("Re-evaluating the corrected code...")
corrected\_evaluation\_results = evaluate\_prime\_function(corrected\_code, test\_cases)
print("\\n--- Corrected Evaluation Results ---")
print(corrected\_evaluation\_results)
print("-----------------------------------\\n")

```

Observe your W&B dashboard again. You'll see another trace for this `call_grok4` and `evaluate_prime_function` pair. Weave allows you to easily compare `evaluation_results` from the initial attempt versus the `corrected_evaluation_results`. This clear comparison demonstrates Grok's iterative improvement and the value of structured evaluation and tracing.

* Grok will typically not only correct the mistake but also explain what was wrong, depending on the prompt. This iterative loop can be repeated: test the new code, and if something else comes up, ask Grok again. This is extremely powerful for troubleshooting edge cases or improving performance.

---

## 5. Leveraging W&B Weave for Deeper Analysis and Workflow Improvement

The power of Weave extends far beyond simple logging. By consistently using `weave.op()` for your interactions and evaluations with Grok 4, you unlock powerful analytical capabilities on your W&B dashboard, turning your code generation process into a measurable and optimizable workflow:

* **Version Control for Prompts & Models:** Every `weave.op` run is recorded as a trace. You can easily navigate through these traces to see how different prompts, context messages, or even versions of Grok 4 (if you were to compare models) affect the generated code and its subsequent evaluation scores. This is crucial for systematic prompt engineering and understanding the impact of your inputs.
* **Cost and Performance Monitoring:** Weave automatically captures metrics like token usage and latency for each `call_grok4` operation (when provided by the API). You can visualize these trends over time to manage costs effectively, identify any performance bottlenecks, and optimize your prompts for efficiency.
* **Failure Analysis and Debugging:** If Grok 4 ever produces "hallucinations," logically flawed code, or simply doesn't meet expectations, you have the exact prompt, response, and evaluation results recorded. This allows you to go back, understand *why* it failed, and iterate on your prompting strategy to improve future outcomes.
* **Comparing Model Variants and Prompts:** Weave's robust logging capabilities enable you to run experiments comparing different prompts for the same task, or even compare the performance of Grok 4 against other LLMs (e.g., `gpt-4o`, `claude-3-opus`) on specific coding benchmarks within a unified dashboard.
* **Automated Dashboards and Reports:** You can create custom, interactive dashboards in W&B to visualize key metrics (e.g., accuracy over time, cost per successful generation, average debugging iterations). These dashboards can be shared with your team, providing transparent insights into the efficiency of your AI-powered development.

By integrating Weave from the outset, the entire process of generating, testing, and debugging code with Grok 4 becomes transparent, measurable, and highly efficient. You're not just getting code; you're building a reliable, observable, and continuously improving AI-powered development workflow.
```
