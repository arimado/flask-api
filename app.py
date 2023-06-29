import os
from flask import Flask, request
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent, load_tools


app = Flask(__name__)

apiKey = os.getenv('OPEN_AI_API_KEY')
os.environ['OPENAI_API_KEY'] = apiKey
serpapi_api_key = os.getenv('SERPAPI_API_KEY')
os.environ['SERPAPI_API_KEY'] = serpapi_api_key
llm = OpenAI(temperature=1.4)
llmSearch = OpenAI(temperature=0)

title_template = PromptTemplate(
    input_variables=['topic'],
    template="The synopsis for a book about {topic}.",
)

title_memory = ConversationBufferMemory(
    input_key='topic', memory_key='chat_history')

title_chain = LLMChain(llm=llm, prompt=title_template,
                       verbose=True, output_key='title', memory=title_memory)

tools = load_tools(["serpapi", "llm-math"], llm=llm)

searchAgent = initialize_agent(
    tools, llmSearch, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# mathAgent = initialize_agent(tools, llm, agent=AgentType.MATH, verbose=true);
# chatAgent = initialize_agent(tools, llm, agent=AgentType.CHAT, verbose=true);
# reactAgent = initialize_agent(tools, llm, agent=AgentType.REACT_DESCRIPTION, verbose=true);

conversation = ConversationChain(llm=llm, verbose=True)

stores = [
    {
        "name": "My Wonderful Store",
        "items": [
            {
                "name": "My Item",
                "price": 15.99
            }
        ]
    }
]


@app.post("/chat")
def chat():
    request_data = request.get_json()
    message = request_data["message"]
    response = conversation.run(message)
    return {"response": response}


@app.post("/search")
def search():
    # get search pramater from request
    search = request.args.get('query')
    result = searchAgent.run(search)
    return {"result": result}


@app.post("/title")
def title():
    request_data = request.get_json()
    topic = request_data["topic"]
    title = title_chain.run(topic)
    return {"title": title}


@app.get("/store")
def get_stores():
    return {"stores": stores}


@app.post("/store")
def create_store():
    request_data = request.get_json()
    new_store = {
        "name": request_data["name"],
        "items": []
    }
    stores.append(new_store)
    return new_store, 201


@app.post("/store/<string:name>/item")
def create_item_in_store(name):
    request_data = request.get_json()
    for store in stores:
        if store["name"] == name:
            new_item = {
                "name": request_data["name"],
                "price": request_data["price"]
            }
            store["items"].append(new_item)
            return new_item, 201
    return {"message": "store not found"}, 404


@app.get("/store/<string:name>")
def get_store(name):
    for store in stores:
        if store["name"] == name:
            return store
    return {"message": "store not found"}, 404
