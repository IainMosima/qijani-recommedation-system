import operator
from typing import List, Annotated, TypedDict
from typing import Optional, Literal

from langgraph.graph import add_messages, MessagesState
from pydantic import BaseModel, Field


class Analyst(BaseModel):
    name: str
    tone: str
    theme: str
    description: str


class MealTypeAnalysts(BaseModel):
    meal_type: str
    analysts: List[Analyst]


class AnalystMessages(MessagesState):
    analyst: Analyst


class SourcedDocuments(TypedDict):
    analyst: Analyst
    context: Annotated[list, operator.add]


class RecommendedMeal(BaseModel):
    meal_name: str
    meal_type: str
    ingredients: list[str]
    preparation_steps: list[str]
    prepared_steps: list[str]
    prep_time_minutes: int
    portion: str
    goal_support: str


class RecommendedMealList(BaseModel):
    meals: List[RecommendedMeal]


class UserProfile(BaseModel):
    name: Optional[str] = Field(None, description="User's full name")
    age: int = Field(..., ge=0, le=120, description="User's age")
    gender: str = Field(..., description="Male, Female, or Other")
    height_cm: int = Field(..., gt=50, lt=250, description="Height in cm")
    weight_kg: float = Field(..., gt=20, lt=300, description="Weight in kg")
    activity_level: str = Field(...,
                                description="Sedentary, Lightly active, Moderately active, Very active, Super active")
    dietary_preferences: List[Literal[
        "Vegetarian", "Vegan", "Pescatarian", "Keto", "Paleo",
        "Gluten-Free", "Dairy-Free", "Nut-Free", "Halal", "Kosher",
        "Low-Carb", "Low-Fat", "High-Protein", "Mediterranean", "FODMAP", "Sugar-Free"
    ]] = Field(default=[], description="User's dietary preferences, can be one or more.")
    allergies: List[str] = Field(default=[], description="User's allergies")
    health_conditions: List[str] = Field(default=[], description="Any medical conditions")
    weight_goal: str = Field(..., description="Lose weight, Maintain weight, Gain muscle")
    past_meals: List[str] = Field(default=[], description="Past meals")


class RecommendedMeals(TypedDict):
    user_profile: UserProfile
    recommended_meals: Annotated[list[RecommendedMealList], add_messages]


class MealQuery(TypedDict):
    meal_type: Literal["BREAKFAST", "LUNCH", "DINNER", "SNACKS"]
    query: str


class InterviewState(BaseModel):
    user_profile: UserProfile
    max_num_turns: int = 1
    web_documents: Annotated[list[SourcedDocuments], operator.add] = Field(default_factory=list)
    wiki_documents: Annotated[list[SourcedDocuments], operator.add] = Field(default_factory=list)
    sourced_documents: Annotated[list[SourcedDocuments], operator.add] = Field(default_factory=list)
    meal_type_analysts: List[MealTypeAnalysts] = Field(default_factory=list)
    interview: str = ""
    analysts_messages: List[AnalystMessages] = Field(default_factory=list)
    recommended_meals: RecommendedMeals = Field(default_factory=dict)
    analysts: List[MealTypeAnalysts] = Field(default_factory=list)
    max_analysts: int = 1
    meal_queries: List[MealQuery] = Field(default_factory=list)


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")
