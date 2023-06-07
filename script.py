import gradio as gr
import pickle
import modules.shared as shared
from modules.extensions import apply_extensions
from modules.text_generation import encode, get_max_prompt_length
from modules.text_generation import generate_reply
from modules.text_generation import generate_reply_wrapper
from modules.text_generation import stop_everything_event
from modules.ui import list_interface_input_elements
from modules.ui import gather_interface_values
from modules.html_generator import generate_basic_html

try:
    with open('notebook.sav', 'rb') as f:
        params = pickle.load(f)
except FileNotFoundError:
    params = {
        "display_name": "Playground",
        "is_tab": True,
        "usePR": False,
        "pUSER": 'USER:',
        "pBOT": 'ASSISTANT:'
    }

def get_last_line(string):
    lines = string.splitlines()
    if lines:
        last_line = lines[-1]
        return last_line
    else:
        return ""


def input_modifier(string):

    modified_string = string
    addLineReply = ""

    if params['usePR']:
        if "---" in string:
            lines = string.splitlines()  # Split the text into lines
            modified_string = ""
            for i, line in enumerate(lines):
                if addLineReply:
                    line.lstrip()
                    line = addLineReply + line
                    addLineReply = ""
                elif line.startswith("---"):
                    line = line.replace("---", params['pUSER'])  
                    addLineReply = params['pBOT'] 
                    
                modified_string = modified_string+ line +"\n"

            if addLineReply:
                modified_string = modified_string + addLineReply

    return modified_string


    
def output_modifier(string):
    #print(f"output_modifier: {string}") 
    return string

def copynote(string):
    return string

def formatted_outputs(reply):
 
    return reply, generate_basic_html(reply)


def generate_reply_wrapperMY(question, state):

    # if use quick prompt, add \n if none
    if params['usePR']:
        if not question.endswith("\n"):
            lastline = get_last_line(question)
            if lastline.startswith("---"):
                question+="\n"

    for reply in generate_reply(question, state, eos_token = None, stopping_strings=None, is_chat=False):
        if shared.model_type not in ['HF_seq2seq']:
            reply = question + reply

        yield formatted_outputs(reply)

def ui():
    #input_elements = list_interface_input_elements(chat=False)
    #interface_state = gr.State({k: None for k in input_elements})
    global addLR
    addLR = False
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Tab('Text'):
                    with gr.Row():
                        text_boxA = gr.Textbox(value='', elem_classes="textbox", lines=20, label = 'Notebook A')
                with gr.Tab('HTML'):
                    with gr.Row():
                        htmlA = gr.HTML()
            with gr.Row():    
                generate_btnA = gr.Button('Generate', variant='primary', elem_classes="small-button")
                stop_btnA = gr.Button('Stop', elem_classes="small-button")
                toNoteB = gr.Button('Copy to B', elem_classes="small-button")
            with gr.Row():
                with gr.Box():
                    with gr.Column():    
                        usePR = gr.Checkbox(value=params['usePR'], label='Enable Quick Instruct (line starts with three dashes --- )')
                        with gr.Row():
                            preset_type = gr.Dropdown(label="Preset", choices=["Custom", "Vicuna", "Alpaca", "Guanaco"], value="Custom")
                            text_USR = gr.Textbox(value=params['pUSER'], lines=1, label='User string')
                            text_BOT = gr.Textbox(value=params['pBOT'], lines=1, label='Bot string')
                        gr.Markdown('Example: --- What are synonyms for happy?')    

        with gr.Column():
            with gr.Row():
                with gr.Tab('Text'):
                    with gr.Row():
                        text_boxB = gr.Textbox(value='', elem_classes="textbox", lines=20, label = 'Notebook B')
                with gr.Tab('HTML'):
                    with gr.Row():
                        htmlB = gr.HTML()
            with gr.Row():    
                generate_btnB = gr.Button('Generate', variant='primary', elem_classes="small-button")
                stop_btnB = gr.Button('Stop', elem_classes="small-button")
                toNoteA = gr.Button('Copy to A', elem_classes="small-button")
            with gr.Row():    
                gr.Markdown('    (v.0.9    FPham 2023)')

   
        #shared.gradio['Undo'] = gr.Button('Undo', elem_classes="small-button")
        #shared.gradio['Regenerate'] = gr.Button('Regenerate', elem_classes="small-button")
    # Todo:
    # add silider for independend temperature, top_p and top_k
    # shared.input_elements, shared.gradio['top_p'] 
    input_paramsA = [text_boxA,shared.gradio['interface_state']]
    output_paramsA =[text_boxA, htmlA]
    input_paramsB = [text_boxB,shared.gradio['interface_state']]
    output_paramsB =[text_boxB, htmlB]
   
    generate_btnA.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        generate_reply_wrapperMY, inputs=input_paramsA, outputs=output_paramsA, show_progress=False)
    stop_btnA.click(stop_everything_event, None, None, queue=False)
    generate_btnB.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        generate_reply_wrapperMY, inputs=input_paramsB, outputs=output_paramsB, show_progress=False)
    stop_btnB.click(stop_everything_event, None, None, queue=False)
    toNoteB.click(copynote, text_boxA, text_boxB)    
    toNoteA.click(copynote, text_boxB, text_boxA)
    
    def update_activate(x):
        params.update({"usePR": x})
        with open('notebook.sav', 'wb') as f:
            pickle.dump(params, f)
    
    def update_stringU(x):
        params.update({"pUSER": x})
        with open('notebook.sav', 'wb') as f:
            pickle.dump(params, f)
    def update_stringB(x):
        params.update({"pBOT": x})
        with open('notebook.sav', 'wb') as f:
            pickle.dump(params, f)
    
    def update_preset(x):
        if x == "Vicuna":
            return 'USER:','ASSISTANT:'
        elif x == "Alpaca":
            return '### Instruction:','### Response:'
        elif x == "Guanaco":
            return '### Human:','### Assistant:'
        
        return 'USER:','ASSISTANT:'           


    usePR.change(update_activate, usePR, None)   
    text_USR.change(update_stringU, text_USR, None) 
    text_BOT.change(update_stringB, text_BOT, None) 
    preset_type.change(update_preset,preset_type,[text_USR,text_BOT])

