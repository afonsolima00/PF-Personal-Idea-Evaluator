import pandas as pd
import json
import google.generativeai as genai
import os
import google.api_core.exceptions

# Configure API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = input("Please enter your Google AI API Key: ")
genai.configure(api_key=GOOGLE_API_KEY)

# Set up the model - Changed to Gemini 1.5 Flash (Free Tier)
try:
    model = genai.GenerativeModel('models/gemini-1.5-flash')  # Use a free-tier compatible model
    print("Using model: models/gemini-1.5-flash")
except google.api_core.exceptions.NotFound as e:
    print(f"Error: {e}")
    print("It seems like this model is not available right now.")
    exit()

# Define multiple examples for few-shot learning
example1_idea = "Fitness Tracker App"
example1_description = "An app to track workouts and fitness goals"
example1_evaluation = '{"viability": "High", "time_estimate": "3 months", "monetization": "Subscription"}'

example2_idea = "Simple Calculator"
example2_description = "A basic calculator with arithmetic operations"
example2_evaluation = '{"viability": "Low", "time_estimate": "1 week", "monetization": "Free"}'

# Function to create a detailed prompt with multiple examples
def create_prompt(idea, description):
    return (
        f"Here are two examples of project idea evaluations:\n\n"
        f"1. For the project idea '{example1_idea} - {example1_description}', "
        f"the evaluation is {example1_evaluation}.\n"
        f"2. For the project idea '{example2_idea} - {example2_description}', "
        f"the evaluation is {example2_evaluation}.\n\n"
        f"Now, evaluate the following project idea: '{idea} - {description}'. "
        f"Provide a unique evaluation in the same JSON format, "
        f"with specific values for 'viability' (e.g., 'High', 'Medium', 'Low'), "
        f"'time_estimate' (e.g., '1 week', '1 month', '3 months'), and "
        f"'monetization' (e.g., 'Subscription', 'Free', 'Ads') based on the given idea and description.\n\n"
        f"Return ONLY the JSON response, nothing else."
    )

# Function to interact with the Gemini API
def get_gemini_response(prompt, model):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        raise

# Load project ideas from CSV
ideas_df = pd.read_csv('ideas.csv')

# List to store evaluation results
results = []

# Process each project idea
for index, row in ideas_df.iterrows():
    idea = row['Idea']
    description = row['Description']

    # Generate the prompt
    prompt = create_prompt(idea, description)
    
    # Get the AI's response
    try:
        response = get_gemini_response(prompt, model)

        # Improved JSON extraction and parsing
        try:
            insights = json.loads(response.strip())
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON directly for idea: {idea}. Attempting extraction.")
            start_index = response.find('{')
            end_index = response.rfind('}')
            if start_index == -1 or end_index == -1 or start_index >= end_index:
                raise ValueError("No valid JSON brackets found.")
            json_str = response[start_index:end_index + 1]
            insights = json.loads(json_str)
        
        results.append({
            'idea': idea,
            'description': description,
            **insights
        })
        print(f"Successfully evaluated: {idea}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON for idea: {idea}. Error: {e}")
        results.append({
            'idea': idea,
            'description': description,
            'viability': 'JSONDecodeError',
            'time_estimate': 'JSONDecodeError',
            'monetization': 'JSONDecodeError'
        })
    except ValueError as e:
        print(f"No valid JSON brackets found for idea: {idea}. Error: {e}")
        results.append({
            'idea': idea,
            'description': description,
            'viability': 'BracketError',
            'time_estimate': 'BracketError',
            'monetization': 'BracketError'
        })
    except Exception as e:  # Catch other potential errors during processing
        print(f"An unexpected error occurred for idea: {idea}. Error: {e}")
        results.append({
            'idea': idea,
            'description': description,
            'viability': 'UnexpectedError',
            'time_estimate': 'UnexpectedError',
            'monetization': 'UnexpectedError'
        })

# Save results to a JSON file
with open('evaluated_ideas.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Evaluation complete. Results saved to 'evaluated_ideas.json'.")
