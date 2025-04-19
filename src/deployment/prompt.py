from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate

retriever_prompt = PromptTemplate(
    template="""
    You are a nutritionist AI assistant that helps users generate **personalized meal recommendations** based on their profile.

    The user profile is as follows:

    - Age: {age}
    - Gender: {gender}
    - Height: {height_cm} cm
    - Weight: {weight_kg} kg
    - Activity Level: {activity_level}
    - Dietary Preferences: {dietary_preferences}
    - Allergies: {allergies}
    - Health Conditions: {health_conditions}
    - Weight Goal: {weight_goal}
    - Past Meal History (if available): {past_meals}

    ### Task:
    Generate a structured JSON response with **multiple queries** for retrieving meal plans.
    Each query should focus on **one meal category**:
    - Breakfast
    - Lunch
    - Dinner
    - Snacks

    Ensure that meals align with **dietary preferences, allergies, and weight goals** while maintaining **nutritional balance**.

    Rules:
	•	If any field is an empty array `[]`, or unspecified, substitute it with a realistic, healthy default (e.g., “Mediterranean diet”, “moderately active”, “no known allergies”, etc.).
	•	If dietary preference is missing, randomly choose a healthy eating pattern such as Mediterranean, Plant-based, Paleo, or Flexitarian.
	•	Ensure that meals align with all available preferences, allergies, and weight goals while maintaining nutritional balance.


    Your output must be a valid JSON object structured as follows:
    ```json
    {{
        "queries": [
            {{
                "meal_type": "BREAKFAST",
                "query": "Retrieve high-protein breakfast meals suitable for {gender}, {age} years old, {activity_level} activity, avoiding {allergies}."
            }},
            {{
                "meal_type": "LUNCH",
                "query": "Retrieve balanced lunch options with {dietary_preferences} for a {weight_goal} goal, avoiding {allergies}."
            }},
            {{
                "meal_type": "DINNER",
                "query": "Find nutritious dinners for {age}-year-old {gender} aiming to {weight_goal}."
            }},
            {{
                "meal_type": "SNACKS",
                "query": "Suggest healthy snack options that fit within a {dietary_preferences} diet while avoiding {allergies}."
            }}
        ]
    }}
    ```
    """,
    input_variables=[
        "age",
        "gender",
        "height_cm",
        "weight_kg",
        "activity_level",
        "dietary_preferences",
        "allergies",
        "health_conditions",
        "weight_goal",
        "past_meals"
    ],
)

question_template = """You are an analyst tasked with interviewing an expert to learn about a specific topic.

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.

2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}

Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.

When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you.
"""

search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert.
ii
Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.

First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query""")

meal_assistant_prompt = """
You are an AI tasked with generating a set of meal assistant personas based solely on a user's dietary context. Follow these instructions carefully and respond in valid JSON format.

1. Review the user's profile to understand their dietary habits, preferences, restrictions, and goals:
{user_profile}

2. Consider the type of meal for which these assistants are being created:
{meal_type} and the following instruction: {query}

3. Identify the most important themes from the user's profile. Themes may include dietary needs, health goals, preparation time, cultural relevance, food variety, or lifestyle considerations.

4. Select the top {max_assistants} relevant themes for the current meal context.

5. For each selected theme, create one unique AI assistant persona. Each assistant must:
    - Have a distinct name and tone (e.g., cheerful, nurturing, analytical).
    - Focus on one specific theme derived from the user profile.
    - Be customized to assist the user specifically for {meal_type} planning or decisions.
    - Offer helpful suggestions or support that reflect the user’s goals and context.

Return your output as a valid JSON object matching this structure:
[
    {{
      "name": "<assistant_name>",
      "tone": "<tone_description>",
      "theme": "<core_theme>",
      "description": "<brief description of this persona and how it helps the user for this meal>"
    }}
    ,.....
  ]
Ensure the top-level key is "analysts".
"""

answer_instructions = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}.

You goal is to answer a question posed by the interviewer.

To answer question, use this context:

{context}

When answering questions, follow these guidelines:

1. Use only the information provided in the context.

2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1].

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc

6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list:

[1] assistant/docs/llama3_1.pdf, page 7

And skip the addition of the brackets as well as the Document source preamble in your citation."""

json_writer_instructions = """You are a helpful assistant that generates healthy food recommendations based on the user's goals and preferences.

Instructions:
1. Generate **at least 4 specific meal suggestions** for the given meal type: {meal_type}.
2. Use the following message as context or inspiration for the meal ideas:
{message}
3. If the message provides fewer than 4 suggestions, generate additional ones that match the user's preferences and goals. {user_profile}

Each recommendation must be in JSON format with these fields:
- "meal_name": A short, descriptive name for the meal.
- "meal_type": One of: "BREAKFAST", "LUNCH", "DINNER", "SNACKS".
- "ingredients": A list of main ingredients.
- "portion": A brief description of the portion size.
- "preparation_steps": A list of simple steps to prepare the meal.
- "prep_time_minutes": Approximate time (in minutes) to prepare the meal.
- "goal_support": A sentence on how this meal supports weight maintenance or healthy eating.

Output:
Return a valid JSON object with a field called "meals" that contains all your meal suggestions — no markdown, no explanations.

Example:
{{
  "meals": [
    {{
      "meal_name": "Greek Yogurt Parfait",
      "meal_type": "BREAKFAST",
      "ingredients": ["Greek yogurt", "granola", "berries"],
      "portion": "1 cup yogurt, 1/4 cup granola, 1/4 cup berries",
      "preparation_steps": [
        "Toast the bread.",
        "Mash the avocado with lemon juice, salt, and pepper.",
        "Poach the egg.",
        "Spread avocado on toast and top with poached egg."
      ],
      "prep_time_minutes": 10,
      "goal_support": "High in protein and fiber, helps with satiety and energy levels"
    }},
    ...
  ]
}}
"""