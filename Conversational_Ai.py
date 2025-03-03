import openai #for GPT-based chatbot functionality
import argparse  # Import argparse for command-line argument parsing
import base64  # Import base64 for encoding image files
from flask import Flask, request, jsonify, render_template  # Import Flask for web application
from flask_cors import CORS  # Import CORS for handling cross-origin requests
import fitz  # PyMuPDF - Library for handling PDFs
from io import BytesIO  # Import BytesIO for handling binary file operations
from langchain.memory import ConversationBufferMemory  # Import memory buffer for conversation history
from langchain.schema import HumanMessage, AIMessage  # Import message schema from LangChain
import logging  # Import logging for debugging and information logging

# Setting OpenAI API key (This should be securely stored in an environment variable in production)
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY  # Assign API key to OpenAI client

# Initialize Flask application
app = Flask(__name__)  # Create Flask instance
CORS(app, resources={r"/*": {"origins": "*"}})  
# Enable CORS to allow cross-origin requests

# Initialize global conversation memory
memory = ConversationBufferMemory(return_messages=True)  # Store conversation history

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)  # Configure logging level to INFO
logger = logging.getLogger(__name__)  # Create logger instance

def extract_text_from_pdf(pdf_file):
    """Extracts text content from a given PDF file using PyMuPDF (fitz)."""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")  # Open PDF file from binary stream
        text = "\n".join([page.get_text("text") for page in pdf_document])  # Extract text from each page
        return text  # Return extracted text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"  # Return error message if extraction fails

def moderate_content(response_text):
    """Moderates AI-generated content to filter inappropriate words."""
    prohibited_words = ["hate speech", "violence", "discrimination"]  # List of prohibited words
    for word in prohibited_words:
        if word in response_text.lower():  # Check if response contains prohibited words
            return "I'm sorry, but I can't provide a response to that request."  # Return a filtered response
    return response_text  # Return response if no prohibited words are found

def chat_with_gpt4(user_input, image_file=None, pdf_file=None):
    """Handles user input, including text, images, and PDFs, and interacts with GPT-4-Turbo."""
    try:
        pdf_text = None  # Initialize PDF text variable
        image_data = None  # Initialize image data variable

        if pdf_file:
            pdf_text = extract_text_from_pdf(pdf_file)  # Extract text from PDF
            if pdf_text.startswith("Error"):  # If extraction fails, return error message
                return pdf_text
        elif image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")  # Encode image as base64
        
        # Load past conversation history from memory
        history = memory.load_memory_variables({})["history"]
        
        # Convert history into OpenAI API format
        messages = [{"role": "system", "content": "You are an AI assistant that remembers past conversations."}]
        for message in history:
            if isinstance(message, HumanMessage):
                messages.append({"role": "user", "content": message.content})  # Add user messages
            elif isinstance(message, AIMessage):
                messages.append({"role": "assistant", "content": message.content})  # Add AI messages
        
        # Add current user input
        if user_input:
            messages.append({"role": "user", "content": user_input})
        if pdf_text:
            messages.append({"role": "user", "content": pdf_text})
        elif image_data:
            messages.append({"role": "user", "content": [{"type": "text", "text": user_input or "Describe the image."},
                                                              {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}]})
        
        # Query OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=350,
            temperature=0.7
        )
        
        # Extract AI response
        gpt_response = response["choices"][0]["message"]["content"].strip()
        
        # Save conversation history
        memory.save_context({"input": user_input or "[File Sent]"}, {"output": gpt_response})
        
        return moderate_content(gpt_response)  # Return moderated AI response
    except Exception as e:
        return f"Error: {str(e)}"  # Return error message if processing fails

# Define web routes
@app.route('/')
def index():
    return render_template('index.html')  # Render HTML template

@app.route('/chat', methods=['POST'])
def chat_api():
    """Handles chat API requests."""
    data = request.form  # Get form data
    user_input = data.get("input")  # Extract user input
    image_file = request.files.get("image")  # Extract uploaded image
    pdf_file = request.files.get("pdf")  # Extract uploaded PDF
    
    if not user_input and not image_file and not pdf_file:  # Check if input is empty
        return jsonify({"error": "No input, image, or PDF provided!"}), 400

    response = chat_with_gpt4(user_input, image_file=image_file, pdf_file=pdf_file)  # Process input
    return jsonify({"response": response})  # Return AI response

@app.route('/reset', methods=['POST'])
def reset_memory():
    """Resets conversation memory."""
    memory.clear()  # Clear stored conversation history
    return jsonify({"message": "Memory reset successfully."})  # Confirm reset

# Run chatbot application
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the chatbot as CLI or Web API")
    parser.add_argument("--mode", choices=["cli", "web"], default="web", help="Run mode: cli (default) or web")
    args, unknown = parser.parse_known_args()
    
    if args.mode == "cli":
        print("\nCLI mode is not supported with global memory. Please use web mode.\nUse: python Conversational_Ai.py --mode web\n")
    else:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)  # Start Flask server



