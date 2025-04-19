import json
from typing import List

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.deployment.prompt import retriever_prompt, meal_assistant_prompt, question_template, search_instructions, \
    answer_instructions, json_writer_instructions
from src.deployment.state import InterviewState, MealQuery, MealTypeAnalysts, SearchQuery, AnalystMessages, \
    SourcedDocuments, RecommendedMealList


def generate_retrieval_queries(state: InterviewState):
    user_profile = state.user_profile
    formatted_prompt = retriever_prompt.format(**user_profile.model_dump())
    llm_chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    llm_chat_json = llm_chat.with_structured_output(MealQuery)
    retrieval_queries = llm_chat_json.invoke(formatted_prompt)

    return {"meal_queries": retrieval_queries["queries"]}



def create_analysts(state: InterviewState):
    llm_chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    llm_chat_json = llm_chat.with_structured_output(method="json_mode")
    max_assistants = state.max_analysts
    meal_queries = state.meal_queries
    user_profile = state.user_profile

    result: List[MealTypeAnalysts] = []

    for meal in meal_queries:
        meal_type = meal["meal_type"]
        query = meal["query"]

        system_message = meal_assistant_prompt.format(
            user_profile=user_profile,
            meal_type=meal_type,
            query=query,
            max_assistants=max_assistants,
        )

        analysts = llm_chat_json.invoke(
            [SystemMessage(content=system_message)] + [HumanMessage(content="Generate the set of analysts.")])

        generated_analyst = MealTypeAnalysts(
            meal_type=meal_type,
            analysts=analysts["analysts"],
        )

        result.append(generated_analyst)

    # Write to state
    return {"meal_type_analysts": result}

def generate_question(state: InterviewState):
    """ Node to generate a question """
    llm_chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    meal_type_analysts: List[MealTypeAnalysts] = state.meal_type_analysts
    analysts_messages = state.analysts_messages

    for meal_type_analyst in meal_type_analysts:
        analysts = meal_type_analyst.analysts
        for analyst in analysts:
            goals = f"Explore expert strategies related to {analyst.theme.lower()} — specifically, {analyst.description}"
            system_message = question_template.replace("{goals}", goals)
            question = llm_chat.invoke([
                SystemMessage(content=system_message),
                HumanMessage(content="Let's begin.")
            ])

            analysts_messages.append(
                AnalystMessages(
                    analyst=analyst,
                    messages=[question],
                )
            )

    # Write messages to state
    return {"analysts_messages": analysts_messages}

def search_web(state: InterviewState):
    """ Retrieve docs from web search """
    llm_chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    tavily_search = TavilySearchResults(max_results=3)


    structured_llm = llm_chat.with_structured_output(SearchQuery)
    analysts_messages: List[AnalystMessages] = state.analysts_messages

    all_sourced_documents = []

    for analyst_message in analysts_messages:
        messages = analyst_message.get("messages")
        analyst = analyst_message.get("analyst")
        search_query = structured_llm.invoke([search_instructions] + messages)
        # Search
        search_docs = tavily_search.invoke(search_query.search_query)
        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                for doc in search_docs
            ]
        )

        all_sourced_documents.append(SourcedDocuments(
            analyst=analyst,
            context=[formatted_search_docs],
        ))

    return {"web_documents": all_sourced_documents}


def search_wikipedia(state: InterviewState):
    """ Retrieve docs from wikipedia """


    llm_chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    structured_llm = llm_chat.with_structured_output(SearchQuery)
    analysts_messages: List[AnalystMessages] = state.analysts_messages
    # sourced_documents: List[SourcedDocuments] = state.sourced_documents

    all_sourced_documents = []

    for analyst_message in analysts_messages:
        messages = analyst_message["messages"]
        analyst = analyst_message.get("analyst")
        search_query = structured_llm.invoke([search_instructions] + messages)
        # Search
        search_docs = WikipediaLoader(query=search_query.search_query,
                                      load_max_docs=2).load()

        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ]
        )

        all_sourced_documents.append(
            SourcedDocuments(
                analyst=analyst,
                context=[formatted_search_docs],
            )
        )

    return {"wiki_documents": all_sourced_documents}

def combine_documents(state: InterviewState):
    """Combine documents from different sources"""
    combined_docs = state.web_documents + state.wiki_documents
    return {"sourced_documents": combined_docs}

def generate_answer(state: InterviewState):
    """ Node to answer a question """
    llm_chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Get state
    sourced_documents: List[SourcedDocuments] = state.sourced_documents
    analysts_messages: List[AnalystMessages] = state.analysts_messages


    for index, analyst in enumerate(analysts_messages):
        messages = analyst["messages"]
        analyst_name = analyst["analyst"].name

        i = next(
            (i for i, analyst_message in enumerate(analysts_messages) if
             analyst_message["analyst"].name == analyst_name),
            -1
        )
        context = list(
            filter(lambda doc: doc["analyst"].name == analyst_name, sourced_documents)
        )

        # Answer question
        system_message = answer_instructions.format(goals=analyst["analyst"].tone, context=context)
        answer = llm_chat.invoke([SystemMessage(content=system_message)] + messages)

        # Name the message as coming from the expert
        answer.name = "expert"
        # Answer the question in the user state
        previous_message = analysts_messages[index]["messages"]
        analysts_messages[index]["messages"] = [*previous_message, answer]

    return {"analysts_messages": analysts_messages}


def route_messages(state: InterviewState,
                   name: str = "expert"):
    """ Route between question and answer """
    analysts_messages: List[AnalystMessages] = state.analysts_messages
    max_num_turns = state.max_num_turns

    for analyst_message in analysts_messages:
        messages = analyst_message.get("messages")

        # Check the number of expert answers
        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )

        if num_responses < max_num_turns:
            return "ask_question"

    return "save_interview"


def save_interview(state: InterviewState):
    """ Save interviews """

    # Get messages
    # messages = state["messages"]
    #
    # # Convert interview to a string
    # interview = get_buffer_string(messages)

    place_holder = "**** Everything is good, will implement save interview soon ****"

    # Save to interviews key
    return {"interview": place_holder}

def write_recommendations(state: InterviewState):
    """ Extract food recommendations as JSON """
    llm_chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    analysts_messages: List[AnalystMessages] = state.analysts_messages
    meal_type_analysts:List[MealTypeAnalysts] = state.meal_type_analysts

    recommended_meals = state.recommended_meals

    ai_recommendation: List[RecommendedMealList] = []

    recommended_meals["user_profile"] = state.user_profile

    def retrieve_meal_type(meal_type_analysts: List[MealTypeAnalysts], index_to_look):
        meal_type_analyst = meal_type_analysts[index_to_look]
        return meal_type_analyst.meal_type

    for index, analyst_message in enumerate(analysts_messages):
        analyst = analyst_message.get("analyst")
        meal_type = retrieve_meal_type(meal_type_analysts, index)

        messages = analyst_message["messages"]
        meal_recommendation_message_content = messages[-1].content

        system_message = json_writer_instructions.format(meal_type=meal_type, message=meal_recommendation_message_content, user_profile=state.user_profile)

        response = llm_chat.invoke([
            SystemMessage(content=system_message)
        ])

        parsed = json.loads(response.content)
        ai_recommendation.extend(parsed.get("meals"))

    recommended_meals["recommended_meals"] = ai_recommendation

    return {"recommended_meals": recommended_meals}

# Add nodes and edges
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("combine_documents", combine_documents)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_recommendations", write_recommendations)
interview_builder.add_node("generate_retrieval_queries", generate_retrieval_queries)
interview_builder.add_node("create_analysts", create_analysts)

# Flow
interview_builder.add_edge(START, "generate_retrieval_queries")
interview_builder.add_edge("generate_retrieval_queries", "create_analysts")
interview_builder.add_edge("create_analysts", "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_wikipedia", "combine_documents")
interview_builder.add_edge("search_web", "combine_documents")
interview_builder.add_edge("combine_documents", "answer_question")

interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question', 'save_interview'])
interview_builder.add_edge("save_interview", "write_recommendations")
interview_builder.add_edge("write_recommendations", END)

interview_graph = interview_builder.compile().with_config(run_name="Conduct Interviews")