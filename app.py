import os
from flask import Flask, request
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


app = Flask(__name__)

apiKey = os.getenv('OPEN_AI_API_KEY')
os.environ['OPENAI_API_KEY'] = apiKey
llm = OpenAI(temperature=1.4)

title_template = PromptTemplate(
    input_variables=['topic'],
    template="The synopsis for a book about {topic}.",
)

title_memory = ConversationBufferMemory(
    input_key='topic', memory_key='chat_history')

title_chain = LLMChain(llm=llm, prompt=title_template,
                       verbose=True, output_key='title', memory=title_memory)

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
