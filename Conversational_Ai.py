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
from dotenv import load_dotenv

load_dotenv()

# Load Azure OpenAI credentials from environment variables
AZURE_OPENAI_API_KEY = "58EiaI9j6BX6VitjblauiH5LSAJyCxMrzKA1XKhgVrF49M5qrODmJQQJ99BCACYeBjFXJ3w3AAABACOGRUrq"
AZURE_OPENAI_ENDPOINT = "https://prashchatbot.openai.azure.com/"
AZURE_DEPLOYMENT_NAME = "gpt-35-turbo"

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_DEPLOYMENT_NAME:
    raise ValueError("Azure OpenAI API Key, Endpoint, or Deployment Name not set.")

openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-03-15-preview"
openai.api_key = AZURE_OPENAI_API_KEY

# Initialize Flask application
app = Flask(__name__)

# Configure CORS more specifically for production
CORS(app, resources={
    r"/*": {
        "origins": ["*", "http://127.0.0.1:5000", "https://prashchatbot-e0a4d6cjekfxgkbe.centralus-01.azurewebsites.net"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

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
        logger.error(f"PDF extraction error: {str(e)}")
        return f"Error extracting text from PDF: {str(e)}"

def moderate_content(response_text):
    prohibited_words = ["hate speech", "violence", "discrimination"]
    for word in prohibited_words:
        if word in response_text.lower():
            return "I'm sorry, but I can't provide a response to that request."
    return response_text

def chat_with_gpt4(user_input, image_file=None, pdf_file=None):
    try:
        logger.info(f"Processing request: text={user_input[:50] if user_input else None}, image={bool(image_file)}, pdf={bool(pdf_file)}")
        
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

        logger.info(f"Sending request to OpenAI with {len(messages)} messages")
        
        response = openai.ChatCompletion.create(
            engine=AZURE_DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=350,
            temperature=0.7
        )
        
        gpt_response = response["choices"][0]["message"]["content"].strip()
        memory.save_context({"input": user_input or "[File Sent]"}, {"output": gpt_response})
        
        logger.info(f"Received response from OpenAI: {gpt_response[:50]}...")
        return moderate_content(gpt_response)
    except Exception as e:
        logger.error(f"Error in chat_with_gpt4: {str(e)}")
        return f"Error: {str(e)}"


def command_line_chatbot():
    """
    Runs a command-line chatbot that interacts with the user.
    """
    print("AI Assistant: Hello! How can I assist you today? (Type 'exit' to quit or 'upload image' to send an image)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("AI Assistant: Goodbye!")
            break
        
        if user_input.lower() == "upload image":
            image_path = input("Enter the path to the image: ")
            try:
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                ai_response = chat_with_gpt4("", image_data)
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
        else:
            ai_response = chat_with_gpt4(user_input)
        
        print(f"AI Assistant: {ai_response}")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat_api():
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        logger.info(f"Received chat request with content type: {request.content_type}")
        logger.info(f"Form data keys: {list(request.form.keys())}")
        logger.info(f"Files keys: {list(request.files.keys())}")
        
        user_input = request.form.get("input", "")
        image_file = request.files.get("image")
        pdf_file = request.files.get("pdf")
        
        if not user_input and not image_file and not pdf_file:
            logger.warning("No input provided")
            return jsonify({"error": "No input, image, or PDF provided!"}), 400

        response = chat_with_gpt4(user_input, image_file=image_file, pdf_file=pdf_file)
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in chat_api: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST', 'OPTIONS'])
def reset_memory():
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        memory.clear()
        logger.info("Memory reset successfully")
        return jsonify({"message": "Memory reset successfully."})
    except Exception as e:
        logger.error(f"Error in reset_memory: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the chatbot as CLI or Web API")
    parser.add_argument("--mode", choices=["cli", "web"], default="web", help="Run mode: cli or web")
    args, unknown = parser.parse_known_args()
    
    port = int(os.environ.get("PORT", 8000))
    
    if args.mode == "cli":
        print("\nCLI mode is not supported with global memory. Please use web mode.\nUse: python Conversational_Ai.py --mode web\n")
    else:
        # For production in Azure, use the host and port that Azure provides
        host = os.environ.get("HOST", "0.0.0.0")
        app.run(host=host, port=port)


