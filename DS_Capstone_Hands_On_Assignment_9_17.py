import praw
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from huggingface_hub import login

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
user_agent = os.getenv('USER_AGENT')
token = os.getenv("HUGGING_FACE_TOKEN")


# Log in to Hugging Face using your token
login(token)


# Create Reddit instance with your credentials
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# Load the LLaMA 2 7B Chat model and tokenizer from Hugging Face
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Modify the subreddit and keyword search for your project
subreddit = reddit.subreddit('diabetes')  # Example: Posts related to diabetes
project_posts = subreddit.search('symptoms', limit=5)

posts_data = []
for post in project_posts:
    posts_data.append({
        'title': post.title,
        'selftext': post.selftext,
        'score': post.score,
        'num_comments': post.num_comments,
        'created': post.created_utc
    })

# Print some of the collected posts
for i, post in enumerate(posts_data[:5]):
    print(f"Post {i+1}: {post['title']}\n{post['selftext']}\n{'-'*60}")


# Function to summarize using the LLaMA model
def summarize_post_llama(post_text):
    # Define the prompt for summarization
    prompt = f"Summarize the following post:\n\n{post_text}\n\nSummary:"
    
    # Tokenize and encode the input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the summary using the model
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,  # Adjust max length as needed
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    # Decode the generated text
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.strip()

# Summarize project-relevant posts
for i, post in enumerate(posts_data[:5]):
    summary = summarize_post_llama(post['selftext'])
    print(f"Post {i+1} Summary: {summary}\n{'-'*60}")
