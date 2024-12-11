from IPython.display import clear_output
import subprocess
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# ... [Previous installation code remains the same] ...

import gradio
gradio.strings.en["SHARE_LINK_DISPLAY"] = ""
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from threading import Thread

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = tuple(set(stop_token_ids))
    pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].item() in self.stop_token_ids
    pass
pass

def modify_message_with_pretext(message, pretext):
    """
    Prepend a pretext to the user's message
    """
    return f"{pretext} {message}"

def async_process_chatbot(message, history, pretext=""):
    # Add pretext to the message if provided
    if pretext:
        message = modify_message_with_pretext(message, pretext)
    
    eos_token = tokenizer.eos_token
    stop_on_tokens = StopOnTokens([eos_token,])
    text_streamer  = TextIteratorStreamer(tokenizer, skip_prompt = True)

    # From https://www.gradio.app/guides/creating-a-chatbot-fast
    history_transformer_format = history + [[message, ""]]
    messages = []
    for item in history_transformer_format:
        messages.append({"role": "user",      "content": item[0]})
        messages.append({"role": "assistant", "content": item[1]})
    pass
    # Remove last assistant and instead use add_generation_prompt
    messages.pop(-1)

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda", non_blocking = True)

    # Add stopping criteria - will not output EOS / EOT
    generation_kwargs = dict(
        input_ids = input_ids,
        streamer = text_streamer,
        max_new_tokens = 1024,
        stopping_criteria = StoppingCriteriaList([stop_on_tokens,]),
        temperature = 0.7,
        do_sample = True,
    )
    thread = Thread(target = model.generate, kwargs = generation_kwargs)
    thread.start()

    # Yield will save the output to history!
    generated_text = ""
    for new_text in text_streamer:
        if new_text.endswith(eos_token):
            new_text = new_text[:len(new_text) - len(eos_token)]
        generated_text += new_text
        yield generated_text
    pass
pass

studio_theme = gradio.themes.Soft(
    primary_hue = "teal",
)

# Define pretext buttons
def create_pretext_button(pretext):
    return gradio.Button(value=f"Pretext: {pretext}")

# Create three different pretext buttons
code_pretext_btn = create_pretext_button("Act as a Python code expert")
professional_pretext_btn = create_pretext_button("Respond professionally")
creative_pretext_btn = create_pretext_button("Be highly creative")

scene = gradio.ChatInterface(
    async_process_chatbot,
    chatbot = gradio.Chatbot(
        height = 325,
        label = "Unsloth Studio Chat",
    ),
    textbox = gradio.Textbox(
        placeholder = "Message Unsloth Chat",
        container = False,
    ),
    # Add the pretext buttons to the interface
    additional_inputs = [
        code_pretext_btn,
        professional_pretext_btn,
        creative_pretext_btn,
    ],
    title = None,
    theme = studio_theme,
    examples = None,
    cache_examples = False,
    retry_btn = None,
    undo_btn = "Remove Previous Message",
    clear_btn = "Restart Entire Chat",
)

# Modify the input to pass the pretext
def handle_pretext_button(message, history, evt: gradio.SelectData):
    pretext_map = {
        "Act as a Python code expert": "You are an expert Python programmer. Provide detailed, professional code solutions.",
        "Respond professionally": "Please respond in a clear, concise, and professional manner.",
        "Be highly creative": "Respond with maximum creativity, using imaginative and inspiring language."
    }
    pretext = pretext_map[evt.value.replace("Pretext: ", "")]
    return "", history, pretext

# Add event listeners for the pretext buttons
code_pretext_btn.click(
    fn=handle_pretext_button, 
    inputs=[scene.input_box, scene.chatbot], 
    outputs=[scene.input_box, scene.chatbot, scene.input_box.pretext]
)
professional_pretext_btn.click(
    fn=handle_pretext_button, 
    inputs=[scene.input_box, scene.chatbot], 
    outputs=[scene.input_box, scene.chatbot, scene.input_box.pretext]
)
creative_pretext_btn.click(
    fn=handle_pretext_button, 
    inputs=[scene.input_box, scene.chatbot], 
    outputs=[scene.input_box, scene.chatbot, scene.input_box.pretext]
)

scene.launch(quiet = True)