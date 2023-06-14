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
import json

right_symbol = '\U000027A1'
left_symbol = '\U00002B05'

params = {
        "display_name": "Playground",
        "is_tab": True,
        "usePR": False,
        "pUSER": 'USER:',
        "pBOT": 'ASSISTANT:',
        "selectA": [0,0],
        "selectB": [0,0],
        "max_words": 0,
        "memoryA": 'Bender\'s defining characteristic is his insatiable appetite for vices and mischief. He is often seen smoking cigars, drinking excessive amounts of alcohol (particularly Olde Fortran malt liquor), and engaging in various forms of unethical behavior. He is shamelessly dishonest, frequently stealing, scamming, and manipulating others for personal gain.',
        "memoryB":'',
        "memoryC":'',
        "selectedMEM":'None'
    }

file_nameJSON = "playground.json"


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"



def get_last_line(string):
    lines = string.splitlines()
    if lines:
        last_line = lines[-1]
        return last_line
    else:
        return ""


def input_modifier(string):

    global params
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

    if params['selectedMEM']=='Memory A':
        modified_string = params['memoryA']+'\n'+modified_string  
    elif params['selectedMEM']=='Memory B':
        modified_string = params['memoryB']+'\n'+modified_string 
    elif params['selectedMEM']=='Memory C':
        modified_string = params['memoryC']+'\n'+modified_string 
    

    return modified_string


    
def output_modifier(string):
    #print(f"output_modifier: {string}") 
    return string

def copynote(string):
    return string

def formatted_outputs(reply):
 
    return reply, generate_basic_html(reply)



def generate_reply_wrapperMYSEL(question, state,selectState):

    global params
    selF = params[selectState][0]
    selT = params[selectState][1]
    if not selF==selT:
        print(f"\033[1;32;1m\nGenerarting from selected text in {selectState} and inserting after {params[selectState]}\033[0;37;0m")
        params[selectState] = [0,0]
        before = question[:selF]
        current = question[selF:selT]
        after = question[selT:]
    else:
        current = question
        before = ""
        after = ""
        print(f"\033[1;31;1m\nNo selection in {selectState}, reverting to full text Generate\033[0;37;0m") 
        

    # if use quick prompt, add \n if none
    if params['usePR']:
        if not current.endswith("\n"):
            lastline = get_last_line(current)
            if lastline.startswith("---"):
                current+="\n"

    for reply in generate_reply(current, state, eos_token = None, stopping_strings=None, is_chat=False):
        if shared.model_type not in ['HF_seq2seq']:
            reply = current + reply
        reply = before+reply+after
        yield formatted_outputs(reply)

   

def generate_reply_wrapperMY(question, state, selectState):

    global params
    params[selectState] = [0,0]
    # if use quick prompt, add \n if none
    max_words = int(params['max_words'])

    prepend_str = ''

    if max_words > 0:
        print(f"\033[1;31;1m(Limiting memory to last {max_words} words)\033[0;37;0m")   
        words = question.split(' ')  # Split the question into a list of words
        limited_words = words[-max_words:]  # Get the maximum number of last words
        question = ' '.join(limited_words)  # Join the limited words back into a string
        prepend_str = ' '.join(words[:-max_words])  # Join the words before the maximum words
        if prepend_str:
            prepend_str = prepend_str+ ' '

    if params['usePR']:
        if not question.endswith("\n"):
            lastline = get_last_line(question)
            if lastline.startswith("---"):
                question+="\n"

    for reply in generate_reply(question, state, eos_token = None, stopping_strings=None, is_chat=False):
        if shared.model_type not in ['HF_seq2seq']:
            reply = question + reply

        if prepend_str:
            reply = prepend_str+reply

        yield formatted_outputs(reply)

def ui():
    #input_elements = list_interface_input_elements(chat=False)
    #interface_state = gr.State({k: None for k in input_elements})
    global params

    try:
        with open(file_nameJSON, 'r') as json_file:
            params = json.load(json_file)
    except FileNotFoundError:
        params['max_words'] = 0

    params['selectA'] = [0,0]
    params['selectB'] = [0,0]
    params['selectedMEM']='None'

    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Tab('Text'):
                    with gr.Row():
                        with gr.Column():
                            text_boxA = gr.Textbox(value='', elem_classes="textbox", lines=20, label = 'Notebook A')
                            with gr.Row():
                                with gr.Column(scale=10):
                                    with gr.Row():    
                                        generate_btnA = gr.Button('Generate', variant='primary', elem_classes="small-button")
                                        generate_SelA = gr.Button('Generate [SEL]', variant='primary', elem_classes="small-button")
                                        stop_btnA = gr.Button('Stop', elem_classes="small-button")
                                with gr.Column(scale=1, min_width=50):       
                                    toNoteB = ToolButton(value=left_symbol)                             
                with gr.Tab('HTML'):
                        with gr.Row():
                            htmlA = gr.HTML()
       
            with gr.Row():
                with gr.Box():
                    with gr.Column():    
                        usePR = gr.Checkbox(value=params['usePR'], label='Enable Quick Instruct (line starts with three dashes --- )')
                        with gr.Row():
                            preset_type = gr.Dropdown(label="Preset", choices=["Custom", "Vicuna", "Alpaca", "Guanaco", "OpenAssistant"], value="Custom")
                            text_USR = gr.Textbox(value=params['pUSER'], lines=1, label='User string')
                            text_BOT = gr.Textbox(value=params['pBOT'], lines=1, label='Bot string')
                        gr.Markdown('Example: --- What are synonyms for happy?')    

        with gr.Column():
            with gr.Row():
                with gr.Tab('Text'):
                    with gr.Row():
                        with gr.Column():
                            text_boxB = gr.Textbox(value='', elem_classes="textbox", lines=20, label = 'Notebook B')
                            with gr.Row():
                                with gr.Column(scale=10):
                                    with gr.Row():    
                                        generate_btnB = gr.Button('Generate', variant='primary', elem_classes="small-button")
                                        generate_SelB = gr.Button('Generate [SEL]',variant='primary', elem_classes="small-button")
                                        stop_btnB = gr.Button('Stop', elem_classes="small-button")
                                with gr.Column(scale=1, min_width=50):       
                                    toNoteA = ToolButton(value=left_symbol)                        
                with gr.Tab('HTML'):
                    with gr.Row():
                        htmlB = gr.HTML()
                with gr.Tab('Perma-Memory'):
                    with gr.Column():
                        text_MEMA = gr.Textbox(value=params['memoryA'], lines=5, label='Memory A')
                        text_MEMB = gr.Textbox(value=params['memoryB'], lines=5, label='Memory B')
                        text_MEMC = gr.Textbox(value=params['memoryC'], lines=5, label='Memory C')
                        
                        max_words = gr.Number(label='Limit Memory to last # of words (0 is no limit, 500 is about half page)', value=params['max_words'])
                     
                        with gr.Row():
                            gr.Markdown('Text flow is: Perma-Memory + Limit_Memory ( previous context )') 
                            save_btn = gr.Button('Save', elem_classes="small-button")

            
            with gr.Row():
                with gr.Box():
                    with gr.Column(): 
                        gr_memorymenu = gr.Radio(choices=['None','Memory A','Memory B','Memory C'], value='None', label='Use Perma-Memory', interactive=True)      
                        with gr.Row():                            
                            gr.Markdown('v 6/14/2023 FPHam')    


    selectStateA = gr.State('selectA')
    selectStateB = gr.State('selectB')
        #shared.gradio['Undo'] = gr.Button('Undo', elem_classes="small-button")
        #shared.gradio['Regenerate'] = gr.Button('Regenerate', elem_classes="small-button")
    # Todo:
    # add silider for independend temperature, top_p and top_k
    # shared.input_elements, shared.gradio['top_p'] 
    input_paramsA = [text_boxA,shared.gradio['interface_state'],selectStateA]
    output_paramsA =[text_boxA, htmlA]
    input_paramsB = [text_boxB,shared.gradio['interface_state'],selectStateB]
    output_paramsB =[text_boxB, htmlB]
  
    generate_btnA.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        generate_reply_wrapperMY, inputs=input_paramsA, outputs= output_paramsA, show_progress=False)
    
    generate_SelA.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        generate_reply_wrapperMYSEL, inputs=input_paramsA, outputs=output_paramsA, show_progress=False)

    stop_btnA.click(stop_everything_event, None, None, queue=False)

    toNoteA.click(copynote, text_boxB, text_boxA)

    generate_btnB.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        generate_reply_wrapperMY, inputs=input_paramsB, outputs=output_paramsB, show_progress=False)
    
    generate_SelB.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        generate_reply_wrapperMYSEL, inputs=input_paramsB, outputs=output_paramsB, show_progress=False)

    stop_btnB.click(stop_everything_event, None, None, queue=False)

    toNoteB.click(copynote, text_boxA, text_boxB)    
   

    def on_selectA(evt: gr.SelectData):  # SelectData is a subclass of EventData
        #print (f"You selected {evt.value} at {evt.index} from {evt.target}")
        global params
        params['selectA'] = evt.index
        return ""
    
    def on_selectB(evt: gr.SelectData):  # SelectData is a subclass of EventData
        #print (f"You selected {evt.value} at {evt.index} from {evt.target}")
        global params
        params['selectB'] = evt.index
        return ""


    text_boxA.select(on_selectA, None, None)
    text_boxB.select(on_selectB, None, None)

    def save_pickle():
        global params
        with open(file_nameJSON, 'w') as json_file:
            json.dump(params, json_file,indent=2)
 
    
    def update_activate(x):
        global params
        params.update({"usePR": x})
        save_pickle()
 
    def update_stringU(x):
        global params
        params.update({"pUSER": x})
        save_pickle()

    def update_stringB(x):
        global params
        params.update({"pBOT": x})
        save_pickle()

    def update_max_words(x):
        global params
        params.update({"max_words": x})
        save_pickle()        

    def update_mmemory(A,B,C):
        global params
        params.update({"memoryA": A})
        params.update({"memoryB": B})
        params.update({"memoryC": C})
                
    def update_memorymenu(x):
        global params
        params.update({"selectedMEM": x})

    def update_preset(x):
        if x == "Vicuna":
            return 'USER:','ASSISTANT:'
        elif x == "Alpaca":
            return '### Instruction:','### Response:'
        elif x == "Guanaco":
            return '### Human:','### Assistant:'
        elif x == "OpenAssistant":
            return '<|prompter|>','<|endoftext|><|assistant|>'
        
        return 'USER:','ASSISTANT:'           


    text_MEMA.change(update_mmemory,[text_MEMA,text_MEMB,text_MEMC],None)
    text_MEMB.change(update_mmemory,[text_MEMA,text_MEMB,text_MEMC],None)
    text_MEMC.change(update_mmemory,[text_MEMA,text_MEMB,text_MEMC],None)

    gr_memorymenu.change(update_memorymenu,gr_memorymenu,None)

    usePR.change(update_activate, usePR, None)   
    text_USR.change(update_stringU, text_USR, None) 
    text_BOT.change(update_stringB, text_BOT, None) 
    preset_type.change(update_preset,preset_type,[text_USR,text_BOT])
    max_words.change(update_max_words,max_words,None)
    save_btn.click(save_pickle,None,None)

