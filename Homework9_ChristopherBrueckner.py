import torch
from transformers import pipeline
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TinyLlama text generation pipeline
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16, 
    device_map=device,
)

# Chat template with system and user roles
def format_message(user_message: str):
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in a freindly manner",
        },
        {"role": "user", "content": user_message},
    ]
    # Generate a prompt from the messages using the tokenizer's chat template
    return pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

async def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text("Hello! I am your DSSS_Chris_bot. How can I assist you?")

async def process(update: Update, context: CallbackContext) -> None:
    """Process the user message."""
    inputMsg = update.message.text
    # Format the message to fit TinyLlama's input structure
    prompt = format_message(inputMsg)
    # Generate response using the pipeline
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    # Extract the generated text
    response = outputs[0]["generated_text"]
    # Send the generated response back to the user
    await update.message.reply_text(f"TinyLlama says: {response}")

def main() -> None:
    """Start bot."""
    API_TOKEN = "my_token_here"
    application = Application.builder().token(API_TOKEN).build()
    # Command handler for /start
    application.add_handler(CommandHandler("start", start))
    # Message handler for user messages (excluding commands)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process))
    # Start the bot and keep it running
    application.run_polling()
    

if __name__ == "__main__":
    main()
