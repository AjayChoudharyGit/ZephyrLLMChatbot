import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the Hugging Face model
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def respond(message, history: list[tuple[str, str]], system_message, max_tokens, temperature, top_p):
    # Prepare the messages for the model
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    # Generate the response using the model
    for msg in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = msg.choices[0].delta.content
        response += token
        yield response

# Define the system message
system_message = (
    "You are a Fitness Coach chatbot. Your purpose is to assist users with personalized fitness advice, "
    "workout routines, and progress tracking. You provide guidance on exercise routines, nutrition tips, "
    "and general fitness-related queries. Be motivational and supportive in your responses."
)

# Create the Gradio interface
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value=system_message, label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

if __name__ == "__main__":
    demo.launch()


