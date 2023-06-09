from pathlib import Path
import gradio as gr
from modules import utils
from modules import shared
from modules.models import unload_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import torch
import math
import os

params = {
    "display_name": "Merge",
    "is_tab": True,
}

refresh_symbol = '\U0001f504'  # 🔄

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button

def process_merge(model_name, peft_model_name, output_dir):
    
    base_model_name_or_path = Path(f'{shared.args.model_dir}/{model_name}')
    peft_model_path = Path(f'{shared.args.lora_dir}/{peft_model_name}')
    device_arg = { 'device_map': 'auto' }
    print(f"Unloading model")
    unload_model()

    print(f"Loading base model: {base_model_name_or_path}")
  
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map={'': 0})    
    except Exception as e:
        print(f"\033[91mError in AutoModelForCausalLM: \033[0;37;0m\n {e}" )
        print(f"Merge failed")
        return
    
    print(f"Loading PEFT: {peft_model_path}")
    try:
        model = PeftModel.from_pretrained(base_model, peft_model_path, torch_dtype=torch.float16, device_map={'': 0})
    except Exception as e:
        print(f"\033[91mError initializing PeftModel:  \033[0;37;0m\n{e}")
        print(f"Merge failed")
        return

    print(f"Running merge_and_unload - WAIT untill you see the Model saved message")
    model = model.merge_and_unload()
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    except Exception as e:
        print(f"\033[91mError in AutoTokenizer:  \033[0;37;0m\n{e}")
        print(f"Merge failed")
        return
    model.save_pretrained(f"{output_dir}")
    tokenizer.save_pretrained(f"{output_dir}")
    print(f"Model saved to {output_dir}")
    print(f"**** DONE ****")

def clean_path(base_path: str, path: str):
    """Strips unusual symbols and forcibly builds a path as relative to the intended directory."""
    # TODO: Probably could do with a security audit to guarantee there's no ways this can be bypassed to target an unwanted path.
    # Or swap it to a strict whitelist of [a-zA-Z_0-9]
    path = path.replace('\\', '/').replace('..', '_')
    if base_path is None:
        return path

    return f'{Path(base_path).absolute()}/{path}'



def estimate_proc(raw_text_file, micro_batch_size,batch_size,steps):
    
    epochs = 1
    if raw_text_file not in ['None', '']:
        file_path = clean_path('training/datasets', f'{raw_text_file}.txt')
        file_size = os.path.getsize(file_path)
        #print(f"bytecount {file_size}")
     
        # this is really just made to fit the situation, it's bad math, but I'm too tired
        # one day....
        cutoff_len = 256
        overlap_len = 128
        #epochs = math.ceil((steps * micro_batch_size * (cutoff_len - overlap_len)) / (0.3333333 * file_size))
        epochs =  math.ceil((steps * (128/micro_batch_size) * (cutoff_len - overlap_len)) / ((2.62144 * file_size)))
    
    return str(epochs)

def ui():
    model_name = "None"
    lora_names = "None"
    with gr.Accordion("Merge Model with Lora", open=True):
        
        with gr.Row():
               
            with gr.Column():
                with gr.Row():
                    gr_modelmenu = gr.Dropdown(choices=utils.get_available_models(), value=model_name, label='Model 16-bit HF only')
                    create_refresh_button(gr_modelmenu, lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button')

            with gr.Column():
                with gr.Row():
                    gr_loramenu = gr.Dropdown(multiselect=False, choices=utils.get_available_loras(), value=lora_names, label='LoRA')
                    create_refresh_button(gr_loramenu, lambda: None, lambda: {'choices': utils.get_available_loras(), 'value': lora_names}, 'refresh-button')

        output_dir = gr.Textbox(label='Output Dir', info='The folder name of your merge (relative to text-generation-webui)')
        gr_apply = gr.Button(value='Do Merge')    
        gr_apply.click(process_merge, inputs=[gr_modelmenu, gr_loramenu,output_dir])

    with gr.Accordion("LORA Training Epoch Estimator for plaintext (half-assed)", open=False):
        with gr.Row():
            raw_text_file = gr.Dropdown(choices=utils.get_datasets('training/datasets', 'txt'), value='None', label='Text file', info='The raw text file to use for training.')
            create_refresh_button(raw_text_file, lambda: None, lambda: {'choices': utils.get_datasets('training/datasets', 'txt')}, 'refresh-button')

        with gr.Row():
            micro_batch_size = gr.Slider(label='Micro Batch Size', value=4, minimum=1, maximum=128, step=1, info='Per-device batch size.')
            batch_size = gr.Slider(label='Batch Size', value=128, minimum=0, maximum=1024, step=4, info='Global batch size.', interactive=False )
        

        #with gr.Row():
        #    cutoff_len = gr.Slider(label='Cutoff Length', minimum=0, maximum=2048, value=256, step=32, info='Cutoff length for text input.')
        #    overlap_len = gr.Slider(label='Overlap Length', minimum=0, maximum=512, value=128, step=16, info='Overlap length')

        with gr.Row():
            steps = gr.Slider(label='Desired Max Training steps (around 1500-2000 is usually decent)', minimum=500, maximum=8000, value=2000, step=100, info='Max Steps')
        with gr.Row():
            estimate = gr.Button("Estimate Epochs")
            result = gr.Textbox(label='Epochs to hit Max Steps', value='', info='Estimate number of epochs needed to reach the Max Steps')
       
        estimate.click(estimate_proc, inputs=[raw_text_file, micro_batch_size,batch_size, steps], outputs=result)        

