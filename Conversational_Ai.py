import openai
import argparse
import base64
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import fitz
from io import BytesIO
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import logging

# Load Azure OpenAI credentials from environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_DEPLOYMENT_NAME:
    raise ValueError("Azure OpenAI API Key, Endpoint, or Deployment Name not set.")

openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-03-15-preview"
openai.api_key = AZURE_OPENAI_API_KEY

# Initialize Flask application
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize global conversation memory
memory = ConversationBufferMemory(return_messages=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_file):
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in pdf_document])
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def moderate_content(response_text):
    prohibited_words = ["hate speech", "violence", "discrimination"]
    for word in prohibited_words:
        if word in response_text.lower():
            return "I'm sorry, but I can't provide a response to that request."
    return response_text

def chat_with_gpt4(user_input, image_file=None, pdf_file=None):
    try:
        pdf_text = None
        image_data = None

        if pdf_file:
            pdf_text = extract_text_from_pdf(pdf_file)
            if pdf_text.startswith("Error"):
                return pdf_text
        elif image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        history = memory.load_memory_variables({})["history"]
        messages = [{"role": "system", "content": "You are an AI assistant that remembers past conversations."}]
        for message in history:
            if isinstance(message, HumanMessage):
                messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                messages.append({"role": "assistant", "content": message.content})
        
        if user_input:
            messages.append({"role": "user", "content": user_input})
        if pdf_text:
            messages.append({"role": "user", "content": pdf_text})
        elif image_data:
            messages.append({"role": "user", "content": [{"type": "text", "text": user_input or "Describe the image."},
                                                              {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}]})

        response = openai.ChatCompletion.create(
            engine=AZURE_DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=350,
            temperature=0.7
        )
        
        gpt_response = response["choices"][0]["message"]["content"].strip()
        memory.save_context({"input": user_input or "[File Sent]"}, {"output": gpt_response})
        
        return moderate_content(gpt_response)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_api():
    data = request.form
    user_input = data.get("input")
    image_file = request.files.get("image")
    pdf_file = request.files.get("pdf")
    
    if not user_input and not image_file and not pdf_file:
        return jsonify({"error": "No input, image, or PDF provided!"}), 400

    response = chat_with_gpt4(user_input, image_file=image_file, pdf_file=pdf_file)
    return jsonify({"response": response})

@app.route('/reset', methods=['POST'])
def reset_memory():
    memory.clear()
    return jsonify({"message": "Memory reset successfully."})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the chatbot as CLI or Web API")
    parser.add_argument("--mode", choices=["cli", "web"], default="web", help="Run mode: cli (default) or web")
    args, unknown = parser.parse_known_args()
    
    port = int(os.getenv("PORT", 5000))
    
    if args.mode == "cli":
        print("\nCLI mode is not supported with global memory. Please use web mode.\nUse: python Conversational_Ai.py --mode web\n")
    else:
        app.run(host="0.0.0.0", port=port)




