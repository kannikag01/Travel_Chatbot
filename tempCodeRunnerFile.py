from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

travel_template = """
Determine if the question is related to travel:
Question: {question}
If it is related to travel, answer it. If not, respond with:
"Sorry, I can only assist with travel-related questions. If you have any queries about travel, feel free to ask, and I'll be happy to help!"
Answer:
"""

travel_model = OllamaLLM(model="llama3.2")
travel_prompt = ChatPromptTemplate.from_template(travel_template)
travel_chain = travel_prompt | travel_model


def chatbot_response(context, user_input):
    
    if user_input.lower() in ["hi", "hello"]:
        return "Hi, I am your Travel Bot! How can I assist you today?", context
    elif user_input.lower() == "who are you":
        return "I am your travel guide, here to help you with all your travel-related queries!", context
    elif user_input.lower() == "thank you":
        return "You're welcome! Feel free to ask any further travel-related questions.", context
    elif user_input.lower() == "bye":
        return "Goodbye! Don't hesitate to return if you have more travel-related queries.", context

    

    result = travel_chain.invoke({"context": context, "question": user_input})
 
    context += f"\nUser: {user_input}\nTravel Bot: {result}"
    return result, context


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    context = request.json.get("context", "")
    response, updated_context = chatbot_response(context, user_input)
    return jsonify({"response": response, "context": updated_context})

if __name__ == "__main__":
    app.run(debug=True)
