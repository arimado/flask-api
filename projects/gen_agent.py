from langchain.vectorstores import FAISS
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.chat_models import ChatOpenAI
from langchain.experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory
)
from termcolor import colored
from typing import List
from datetime import datetime, timedelta
import logging
import math
import faiss
logging.basicConfig(level=logging.ERROR)

USER_NAME = "Person A"
LLM = ChatOpenAI(max_tokens=1500)


def relevance_score_fn(score: float) -> float:
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent"""
    # Define embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore,
        other_score_keys=["importance"],
        k=15
    )


tonys_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8
)

tony = GenerativeAgent(
    name="Tony Soprano",
    age=52,
    traits="borderline personality disorder, jokes all the time, likes crime, is the character from the TV Show 'The Sopranos' ",
    status="with mistress",
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=tonys_memory
)

tony_observations = [
    "Tony Soprano remembers his overbearing mother",
    "Tony Soprano gets some money from paulie",
    "Paulie tells Tony that Badabing has been robbed",
    "Tony goes to inspect the club in a rage",
    "The road is noisy at night",
    "Tony Soprano is hungry",
    "Tony Soprano drifts asleep while on the road.",
]

for observation in tony_observations:
    tony.memory.add_memory(observation)

tony_observations_2 = [
    "Tony Soprano wakes up to the sound of his alarm clock ringing.",
    "Tony gets out of bed and heads to the bathroom to freshen up.",
    "Tony realizes he's out of toothpaste and searches through the bathroom cabinet to find a spare tube.",
    "Tony brushes his teeth and takes a quick shower.",
    "Tony dresses in his usual attire, a suit and tie.",
    "Tony goes downstairs to the kitchen and prepares himself a cup of coffee.",
    "Tony sits at the kitchen table and reads the morning newspaper.",
    "Tony's wife, Carmela, joins him at the table for breakfast.",
    "Tony discusses the plans for the day with Carmela and their children.",
    "Tony leaves the house and gets into his black SUV.",
    "Tony drives to his office at the Bada Bing strip club.",
    "Tony meets with his associates and discusses business matters.",
    "Tony receives a phone call from one of his crew members and addresses the issue at hand.",
    "Tony takes a break and enjoys a cigar outside the club.",
    "Tony receives a visit from his therapist, Dr. Melfi, and they have a therapy session.",
    "Tony leaves the office and meets with his consigliere, Silvio Dante, for lunch.",
    "Tony and Silvio discuss ongoing operations and potential business opportunities.",
    "Tony visits a construction site owned by his crew and checks on the progress of the project.",
    "Tony meets with another mob boss to discuss a potential collaboration.",
    "Tony attends a meeting with his capos to discuss the division of profits.",
    "Tony returns home and spends some quality time with his children.",
    "Tony has dinner with his family and shares stories from his day.",
    "Tony watches a baseball game on TV and places bets with his friends.",
    "Tony receives a call from a rival mobster and arranges a meeting to settle a dispute.",
    "Tony spends the evening at a social club, playing cards with his associates.",
    "Tony returns home late at night and goes to bed, ready to face another day in the world of organized crime.",
    "As Tony lies in bed, he suddenly remembers the missing toothpaste earlier. It triggers a nagging suspicion that something isn't quite right. Could it be a clue to a larger problem lurking in his empire?"
]


def start_day():
    for i, observation in enumerate(tony_observations_2):
        _, reaction = tony.generate_reaction(observation)
        print(colored(observation, "green"), reaction)
        if ((i + 1) % 20) == 0:
            print("*" * 40)
            print(
                colored(
                    f"After {i+1} observations, Tommie's summary is:\n{tony.get_summary(force_refresh=True)}",
                    "blue",
                )
            )
            print("*" * 40)


def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message)[1]


def ask_tony(question):
    return interview_agent(tony, question)


def get_faiss():
    print(tony.get_summary(force_refresh=True))
    return "🙈"
