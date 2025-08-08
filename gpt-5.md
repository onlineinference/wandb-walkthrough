# GPT-5 cookbook with W&B Weave observability

This cookbook demonstrates how to get started with GPT-5's advanced features, including the Responses API, reasoning control, structured outputs, and multimodal capabilities, while using W&B Weave for comprehensive observability and experimentation tracking.

## Prerequisites

Before you begin, ensure you have the necessary libraries installed and your accounts set up:

```bash
pip install openai weave wandb pillow pydantic
```

You'll need:
- An OpenAI account and API key with GPT-5 access
- A Weights & Biases account and API key (available from wandb.ai/settings)

## Step 1: Set Up Your OpenAI GPT-5 API Key

Begin by setting up your environment and API key. Choose the method that works best for your environment:

### For Jupyter Notebooks:
```python
# Replace KEY with your actual OpenAI API key
%env OPENAI_API_KEY=KEY
```

### For Python Scripts:
```python
import os

# Method 1: Set directly in code (less secure, but simple)
os.environ['OPENAI_API_KEY'] = "your-actual-api-key-here"

# Method 2: Load from system environment variable
# First set in terminal: export OPENAI_API_KEY="your-key-here"
# Then use: api_key = os.getenv('OPENAI_API_KEY')
```

### For Production Projects (Most Secure):
```python
# Create a .env file with: OPENAI_API_KEY=your-actual-key-here
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
```

### Verify Your Setup:
```python
import os

# Check that your API key is properly set
api_key = os.getenv('OPENAI_API_KEY')
if api_key and api_key.startswith('sk-'):
    print(f"✅ API Key configured: {api_key[:20]}...")
else:
    print("❌ API Key not found. Please check your setup.")
    print("💡 Your API key should start with 'sk-' and be about 51 characters long")
```

**Important Notes:**
- Get your API key from: https://platform.openai.com/account/api-keys
- Never commit API keys to version control
- API keys should start with `sk-` and be approximately 51 characters long
- GPT-5 access may require specific account tiers or early access

## Step 2: Install and Import Required Libraries

Install the necessary packages and import them for use in your notebook or script.

```python
# Import required libraries
import weave
import json
import time
from openai import OpenAI
from typing import Dict, List, Any
import base64
from PIL import Image
import io
from pydantic import BaseModel

print("✅ All libraries imported successfully")
```

## Step 3: Initialize Weave and Configure the OpenAI Client

Set up W&B Weave for observability and configure the OpenAI client for GPT-5 interactions.

```python
# Initialize Weave with a clear project name
# This creates a dedicated space in your W&B dashboard for GPT-5 logs
weave.init('gpt-5-cookbook-tutorial')

# Initialize OpenAI client with your API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Define the GPT-5 model name
GPT5_MODEL = "gpt-5"

print("🟢 Weave initialized and OpenAI client configured")
print("📊 Check your W&B dashboard for the new project: gpt-5-cookbook-tutorial")
```

## Step 4: Create Your First GPT-5 Assistant

Define your custom assistant role and task, just like in the original tutorial but with enhanced observability.

```python
# Get user input for customization (or set directly for automation)
assistant_role = input("Who should I be, as I answer your prompt? ") or "helpful AI assistant"
user_task = input("What do you want me to do? ") or "explain quantum computing in simple terms"

print(f"✅ Assistant Role: {assistant_role}")
print(f"✅ User Task: {user_task}")

# Store these for use throughout the tutorial
gpt_assistant_prompt = f"You are a {assistant_role}"
gpt_user_prompt = user_task
```

## Step 5: Basic Text Generation with the GPT-5 Responses API

Create your first instrumented function to interact with GPT-5's Responses API with observability built-in.

```python
@weave.op()
def generate_with_custom_prompts(
    assistant_prompt: str, 
    user_prompt: str, 
    reasoning_effort: str = "low"
) -> dict:
    """Generate text using your custom prompts with GPT-5's Responses API"""
    
    start_time = time.time()
    
    response = client.responses.create(
        model=GPT5_MODEL,
        reasoning={"effort": reasoning_effort},
        input=[
            {
                "role": "developer",
                "content": assistant_prompt
            },
            {
                "role": "user", 
                "content": user_prompt
            }
        ]
    )
    
    end_time = time.time()
    
    return {
        "output_text": response.output_text,
        "assistant_role": assistant_prompt,
        "user_request": user_prompt,
        "reasoning_effort": reasoning_effort,
        "response_time_seconds": end_time - start_time,
        "model_info": {
            "model_id": GPT5_MODEL,
            "reasoning_tokens": getattr(response, 'reasoning_tokens', None)
        }
    }

# Test your first GPT-5 call
print("🚀 Making your first GPT-5 call...")
basic_result = generate_with_custom_prompts(gpt_assistant_prompt, gpt_user_prompt, "medium")

print("✅ Generated response:")
print(f"Response: {basic_result['output_text'][:200]}...")
print(f"Time taken: {basic_result['response_time_seconds']:.2f} seconds")
```

## Step 6: Access Your W&B Weave Dashboard

After running Step 5, your traces are being logged to W&B Weave even if you don't see clickable links in your output. Here's how to access your dashboard:

```python
print("🔍 Your traces are being logged to W&B Weave!")
print("📊 To view your dashboard:")
print("   1. Go to: https://wandb.ai")
print("   2. Log in to your W&B account")
print("   3. Look for project: 'gpt-5-cookbook-tutorial'")
print("   4. Click on the project to see your traces")
print("")
print("💡 In your dashboard, you'll find:")
print("   - Complete input and output data")
print("   - Response timing and token usage")
print("   - Model parameters and reasoning effort")
print("   - Custom metadata for filtering")
```

**What you'll see in the Weave dashboard:**
- **Traces Tab**: Shows all your GPT-5 function calls
- **Performance Data**: Response times, token usage, reasoning effort levels
- **Input/Output Inspection**: Complete prompts and responses for debugging
- **Custom Metadata**: Searchable tags and parameters for analysis

**Tip**: Bookmark your project URL for easy access throughout the tutorial!

## Step 7: GPT-5 Reasoning Control - Low, Medium, and High Effort Testing

Explore GPT-5's most powerful feature: reasoning effort control across all three levels.

```python
@weave.op()
def generate_with_reasoning_levels(assistant_prompt: str, user_prompt: str) -> dict:
    """Compare your prompts across different reasoning effort levels"""
    results = {}
    efforts = ["low", "medium", "high"]
    
    for effort in efforts:
        print(f"  Generating with {effort} reasoning effort...")
        start_time = time.time()
        
        response = client.responses.create(
            model=GPT5_MODEL,
            reasoning={"effort": effort},
            input=[
                {
                    "role": "developer",
                    "content": assistant_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        
        end_time = time.time()
        
        results[effort] = {
            "output": response.output_text,
            "effort_level": effort,
            "response_time": end_time - start_time,
            "reasoning_tokens": getattr(response, 'reasoning_tokens', None)
        }
    
    return results

# Compare reasoning levels with your custom prompts
print("🧠 Testing all reasoning effort levels...")
reasoning_comparison = generate_with_reasoning_levels(gpt_assistant_prompt, gpt_user_prompt)

print("\n🧠 Your Prompts with Different Reasoning Levels:")
for effort, result in reasoning_comparison.items():
    print(f"\n--- {effort.upper()} EFFORT ({result['response_time']:.2f}s) ---")
    print(result["output"][:300] + "..." if len(result["output"]) > 300 else result["output"])
    print("-" * 50)
```

## Step 8: Structured JSON Output with Your Custom Assistant

Implement GPT-5's breakthrough structured output features using both Pydantic models and JSON schemas.

```python
# Define Pydantic models for structured responses
class AssistantResponse(BaseModel):
    response: str
    key_points: List[str] 
    confidence_level: str
    follow_up_questions: List[str]

# Define JSON schema for structured outputs
ASSISTANT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "response": {
            "type": "string",
            "description": "Main response from the assistant"
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key points or takeaways"
        },
        "confidence_level": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Assistant's confidence in the response"
        },
        "follow_up_questions": {
            "type": "array", 
            "items": {"type": "string"},
            "description": "Suggested follow-up questions"
        }
    },
    "additionalProperties": False,
    "required": ["response", "key_points", "confidence_level", "follow_up_questions"]
}

# Method 1: Using Pydantic models
@weave.op()
def generate_structured_with_pydantic(assistant_prompt: str, user_prompt: str) -> dict:
    """Generate structured output using Pydantic models"""
    try:
        response = client.responses.parse(
            model=GPT5_MODEL,
            input=[
                {"role": "developer", "content": assistant_prompt},
                {"role": "user", "content": user_prompt}
            ],
            text_format=AssistantResponse
        )
        
        return {
            "structured_data": response.output_parsed.model_dump(),
            "raw_output": response.output_text,
            "assistant_role": assistant_prompt,
            "validation_status": "valid",
            "method": "pydantic"
        }
    except Exception as e:
        return {
            "structured_data": None,
            "raw_output": None,
            "validation_status": f"error: {str(e)}",
            "method": "pydantic",
            "error": str(e)
        }

# Method 2: Using JSON schema
@weave.op()
def generate_structured_with_json_schema(assistant_prompt: str, user_prompt: str) -> dict:
    """Generate structured output using JSON schema"""
    try:
        response = client.responses.create(
            model=GPT5_MODEL,
            input=[
                {"role": "developer", "content": assistant_prompt},
                {"role": "user", "content": user_prompt}
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "assistant_response",
                    "strict": True,
                    "schema": ASSISTANT_JSON_SCHEMA
                }
            }
        )
        
        parsed_data = json.loads(response.output_text)
        
        return {
            "structured_data": parsed_data,
            "raw_output": response.output_text,
            "assistant_role": assistant_prompt,
            "validation_status": "valid",
            "method": "json_schema"
        }
    except Exception as e:
        return {
            "structured_data": None,
            "raw_output": getattr(response, 'output_text', None),
            "validation_status": f"error: {str(e)}",
            "method": "json_schema",
            "error": str(e)
        }

# Test both structured output methods
print("📊 Testing Structured Output Methods:")

# Try Pydantic first
print("  --> Testing Pydantic approach...")
pydantic_result = generate_structured_with_pydantic(gpt_assistant_prompt, gpt_user_prompt)
print(f"  ✅ Pydantic: {'Success' if pydantic_result['validation_status'] == 'valid' else 'Failed'}")

# Try JSON schema
print("  --> Testing JSON schema approach...")
schema_result = generate_structured_with_json_schema(gpt_assistant_prompt, gpt_user_prompt)
print(f"  ✅ JSON Schema: {'Success' if schema_result['validation_status'] == 'valid' else 'Failed'}")

# Display successful result
successful_result = pydantic_result if pydantic_result['validation_status'] == 'valid' else schema_result
if successful_result['validation_status'] == 'valid':
    print(f"\n📋 Structured Output (using {successful_result['method']}):")
    print(json.dumps(successful_result['structured_data'], indent=2))
```

## Step 9: Image Generation with GPT-5

Generate images using GPT-5's built-in multimodal capabilities, creating visual content that complements your text.

```python
@weave.op()
def generate_image_with_gpt5(prompt: str) -> dict:
    """Generate an image using GPT-5's built-in image generation"""
    try:
        response = client.responses.create(
            model=GPT5_MODEL,
            input=prompt,
            tools=[{"type": "image_generation"}],
        )
        
        # Extract the image data from the response
        image_data = [
            output.result
            for output in response.output
            if output.type == "image_generation_call"
        ]
        
        if image_data:
            image_base64 = image_data[0]
            
            # Convert base64 to PIL Image for Weave display
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
            
            return {
                "image": image,  # PIL Image object for Weave
                "image_base64": image_base64,
                "prompt_used": prompt,
                "status": "success",
                "generation_successful": True
            }
        else:
            return {
                "image": None,
                "status": "no_image_data",
                "error": "No image data found in response",
                "generation_successful": False
            }
            
    except Exception as e:
        return {
            "image": None,
            "status": "error", 
            "error": str(e),
            "prompt_used": prompt,
            "generation_successful": False
        }

@weave.op()
def create_image_for_existing_content(assistant_prompt: str, existing_content: dict) -> dict:
    """Generate an image based on previously created content"""
    
    # Extract the main response from our structured content
    main_response = existing_content.get("structured_data", {}).get("response", "")
    key_points = existing_content.get("structured_data", {}).get("key_points", [])
    
    if not main_response:
        return {"error": "No existing content found to create image for"}
    
    # Step 1: Have your assistant create an image prompt
    image_prompt_response = client.responses.create(
        model=GPT5_MODEL,
        reasoning={"effort": "low"},
        input=[
            {
                "role": "developer",
                "content": f"{assistant_prompt} Now act as a visual artist and create a detailed image description that would perfectly complement the content you previously created."
            },
            {
                "role": "user",
                "content": f"Create an image description for this content:\n\nMain response: {main_response}\n\nKey points: {', '.join(key_points)}\n\nDescribe an image (under 300 characters) that would enhance this content."
            }
        ]
    )
    
    image_prompt = image_prompt_response.output_text
    print(f"🎨 Generated image prompt: {image_prompt[:100]}...")
    
    # Step 2: Generate the image
    image_result = generate_image_with_gpt5(f"Generate an image: {image_prompt}")
    
    return {
        "original_content": existing_content,
        "image_prompt": image_prompt,
        "image_result": image_result,
        "assistant_role": assistant_prompt,
        "generated_image": image_result.get("image") if image_result.get("generation_successful") else None
    }

# Create an image for your previously generated content
print("🖼️ Creating an image for your content...")

# Use the structured content from Step 8
if 'successful_result' in locals() and successful_result.get("structured_data"):
    print(f"  Using content: '{successful_result['structured_data']['response'][:50]}...'")
    
    multimodal_result = create_image_for_existing_content(gpt_assistant_prompt, successful_result)
    
    if "error" not in multimodal_result:
        print(f"\n📝 Original Content Summary:")
        print(f"  '{multimodal_result['original_content']['structured_data']['response'][:100]}...'")
        
        print(f"\n🎨 Image Concept:")
        print(f"  '{multimodal_result['image_prompt'][:150]}...'")
        
        if multimodal_result["image_result"]["generation_successful"]:
            print(f"✅ Image generated successfully!")
            print("🖼️ Check your Weave dashboard to view the generated image")
        else:
            print(f"❌ Image generation failed: {multimodal_result['image_result']['error']}")
    else:
        print(f"❌ Error: {multimodal_result['error']}")
else:
    print("⚠️ No structured content available. Skipping image generation.")
```

## Step 10: Multi-Step Agentic Workflow

Demonstrate GPT-5's agentic capabilities by orchestrating a complex multi-step workflow with your custom assistant.

```python
@weave.op()
def multi_step_agentic_workflow(
    task_description: str,
    assistant_role: str
) -> dict:
    """
    Demonstrates GPT-5's agentic capabilities through a multi-step workflow.
    Each step is tracked separately for comprehensive observability.
    """
    
    print(f"🤖 Starting agentic workflow: {task_description[:60]}...")
    
    # Step 1: Planning (low effort for quick assessment)
    print("  --> Agent Planning...")
    planning_result = generate_with_custom_prompts(
        assistant_prompt=f"{assistant_role}. Break down complex tasks into logical, actionable steps.",
        user_prompt=f"Create a detailed plan for: {task_description}",
        reasoning_effort="low"
    )
    
    # Step 2: Execution (medium effort for balanced performance)
    print("  --> Agent Executing...")
    execution_result = generate_with_custom_prompts(
        assistant_prompt=f"{assistant_role}. Execute tasks following the provided plan with attention to detail.",
        user_prompt=f"Following this plan:\n{planning_result['output_text']}\n\nNow execute: {task_description}",
        reasoning_effort="medium"
    )
    
    # Step 3: Review and improvement (high effort for thorough analysis)
    print("  --> Agent Reviewing...")
    review_result = generate_with_custom_prompts(
        assistant_prompt=f"{assistant_role}. Provide critical analysis and suggest improvements.",
        user_prompt=f"Review this execution and suggest improvements:\n{execution_result['output_text']}",
        reasoning_effort="high"
    )
    
    total_time = (
        planning_result["response_time_seconds"] + 
        execution_result["response_time_seconds"] + 
        review_result["response_time_seconds"]
    )
    
    return {
        "task_description": task_description,
        "assistant_role": assistant_role,
        "planning_step": planning_result,
        "execution_step": execution_result,
        "review_step": review_result,
        "workflow_completed": True,
        "total_workflow_time": total_time,
        "step_count": 3
    }

# Run a complex agentic workflow
complex_task = f"Building on your role as {assistant_role.split('a ')[-1]}, create a comprehensive strategy for solving: {gpt_user_prompt}"

print(f"🚀 Running multi-step agentic workflow...")
workflow_result = multi_step_agentic_workflow(complex_task, gpt_assistant_prompt)

print(f"\n✅ Workflow completed in {workflow_result['total_workflow_time']:.2f} seconds")
print(f"📊 Steps executed: {workflow_result['step_count']}")

print(f"\n📋 Workflow Summary:")
print(f"  Planning: {workflow_result['planning_step']['output_text'][:100]}...")
print(f"  Execution: {workflow_result['execution_step']['output_text'][:100]}...")
print(f"  Review: {workflow_result['review_step']['output_text'][:100]}...")
```

## Step 11: Comprehensive Tutorial Summary and Analysis

Create a personalized summary using the highest reasoning effort to reflect on your entire GPT-5 journey.

```python
@weave.op()
def create_personalized_tutorial_summary(
    assistant_prompt: str, 
    user_prompt: str,
    session_results: dict
) -> dict:
    """Generate a personalized summary of your GPT-5 tutorial experience"""
    
    summary_request = f"""
    Based on our tutorial session where you were '{assistant_prompt}' and I asked you to '{user_prompt}', 
    please summarize what we accomplished with GPT-5's features:
    
    1. Basic Responses API with custom role prompts
    2. Reasoning effort level comparisons (low/medium/high)
    3. Structured JSON output generation (Pydantic and JSON schema)
    4. Multimodal content creation with image generation
    5. Multi-step agentic workflow orchestration
    6. Advanced observability with W&B Weave logging
    
    Reflect on how your specific role affected the outputs and what developers can learn.
    Provide insights about the reasoning effort trade-offs and structured output reliability.
    """
    
    response = client.responses.create(
        model=GPT5_MODEL,
        reasoning={"effort": "high"},  # Use high effort for comprehensive analysis
        input=[
            {
                "role": "developer", 
                "content": f"{assistant_prompt} Provide a thoughtful, comprehensive summary with actionable insights."
            },
            {
                "role": "user",
                "content": summary_request
            }
        ]
    )
    
    return {
        "personalized_summary": response.output_text,
        "original_assistant_role": assistant_prompt,
        "original_user_request": user_prompt,
        "tutorial_features_completed": [
            "Custom role-based prompting",
            "Responses API usage", 
            "Reasoning effort controls",
            "Structured JSON outputs",
            "Image generation integration",
            "Multimodal content creation",
            "Multi-step agentic workflows",
            "Comprehensive Weave observability"
        ],
        "session_metadata": session_results
    }

# Compile session results
session_summary = {
    "basic_generation": locals().get('basic_result', {}),
    "reasoning_comparison": locals().get('reasoning_comparison', {}),
    "structured_outputs": {
        "pydantic": locals().get('pydantic_result', {}),
        "json_schema": locals().get('schema_result', {})
    },
    "multimodal": locals().get('multimodal_result', {}),
    "agentic_workflow": locals().get('workflow_result', {})
}

print("🎯 Generating comprehensive tutorial summary...")
final_summary = create_personalized_tutorial_summary(
    gpt_assistant_prompt, 
    gpt_user_prompt,
    session_summary
)

print("🎯 Your Personalized GPT-5 Tutorial Summary:")
print("=" * 60)
print(final_summary["personalized_summary"])
print("=" * 60)

print(f"\n📊 Session Details:")
print(f"  Assistant Role: {final_summary['original_assistant_role']}")
print(f"  User Request: {final_summary['original_user_request']}")
print(f"  Features Explored: {len(final_summary['tutorial_features_completed'])}")

print(f"\n✅ Tutorial Complete!")
print(f"🔍 Visit your W&B Weave dashboard to explore all {len([k for k in locals() if 'result' in k])} logged operations")
```

## Step 12: Reviewing Your Complete Work in Weave

After completing all steps, your Weave dashboard contains a comprehensive laboratory notebook of your GPT-5 experimentation.

```python
print("📊 Your W&B Weave Dashboard Now Contains:")
print("""
🔍 Operation Traces:
   - Basic GPT-5 generation calls
   - Reasoning effort comparisons (low/medium/high)
   - Structured output attempts (Pydantic vs JSON schema)
   - Image generation operations
   - Multi-step agentic workflow execution
   - Comprehensive tutorial summary

📈 Performance Metrics:
   - Response times across reasoning effort levels
   - Token usage and computational costs
   - Success rates for structured outputs
   - Image generation success tracking
   - End-to-end workflow timing

🎯 Analysis Capabilities:
   - Side-by-side reasoning effort comparisons
   - Structured output validation tracking
   - Multimodal content generation monitoring
   - Error analysis and debugging insights
   - Custom metadata filtering and grouping

💡 Optimization Insights:
   - Reasoning effort cost-benefit analysis
   - Structured output method reliability
   - Prompt engineering effectiveness
   - Workflow step performance bottlenecks
""")

print("🎉 Congratulations! You've completed the comprehensive GPT-5 cookbook.")
print("🚀 You now have a complete observability setup for production GPT-5 applications.")

# Final statistics
total_operations = len([k for k in locals() if 'result' in k and isinstance(locals()[k], dict)])
print(f"\n📊 Final Stats: {total_operations} operations logged to Weave")
print("🔗 Click any Weave link above to explore your complete GPT-5 experimentation dataset!")
```

This cookbook provides a comprehensive, step-by-step journey through GPT-5's capabilities while building a complete observability foundation with W&B Weave. Each step builds upon the previous one, creating a rich dataset for analysis and optimization of your AI workflows.
