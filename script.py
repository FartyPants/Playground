import gradio as gr
import modules.shared as shared
from modules.extensions import apply_extensions
from modules.text_generation import encode, get_max_prompt_length
from modules.text_generation import generate_reply
from modules.text_generation import generate_reply_wrapper
from modules.text_generation import stop_everything_event
from modules.ui import list_interface_input_elements
from modules.ui import gather_interface_values
from modules.html_generator import generate_basic_html
from pathlib import Path
from modules.LoRA import add_lora_to_model
import re
import json
import os

right_symbol = '\U000027A1'
left_symbol = '\U00002B05'
refresh_symbol = '\U0001f504'  # 🔄


defaultTemp_keys = ['summary_turn', 'summary_include_turn', 'summary_include_turn2']

defaultTemp = {

       "summary_turn":'<|user|> Summarise the following story:\\n\\n<|context|>\\n\\n<|bot|> Summary:\\n\\n',
       "summary_include_turn":'STORY BACKGROUND\\n\\n<|summary|>\\n\\nSTORY\\n\\n',
       "summary_include_turn2":'STORY BACKGROUND\\n\\n<|summary|>\\n\\n<|memory|>\\n\\nSTORY\\n\\n',


}

paraph_undo = ''
paraph_undoSEL = [0,0]
paraph_redo = ''
paraph_redoSEL = [0,0]

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
        "selectedMEM":'None',
        "selectedSUM":'None',
        "summary_turn":'',
        "summary_include_turn":'',
        "summary_include_turn2":'',
        "text_Summary":'',
        "paraph_templ_sel":'Basic',
        "paraph_templ_text":'',
        "paraph_temperament":'Strict'
    }

file_nameJSON = "playground.json"

default_req_params = {
    'max_new_tokens': 200,
    'temperature': 0.7,
    'top_p': 0.1,
    'top_k': 40,
    'repetition_penalty': 1.18,
    'encoder_repetition_penalty': 1.0,
    'suffix': None,
    'stream': True,
    'echo': False,
    'seed': -1,
    'truncation_length': 2048,
    'add_bos_token': True,
    'do_sample': True,
    'typical_p': 1.0,
    'epsilon_cutoff': 0,  # In units of 1e-4
    'eta_cutoff': 0,  # In units of 1e-4
    'tfs': 1.0,
    'top_a': 0.0,
    'min_length': 0,
    'no_repeat_ngram_size': 0,
    'num_beams': 1,
    'penalty_alpha': 0.0,
    'length_penalty': 1,
    'early_stopping': False,
    'mirostat_mode': 0,
    'mirostat_tau': 5,
    'mirostat_eta': 0.1,
    'ban_eos_token': False,
    'skip_special_tokens': True,
    'custom_stopping_strings': [],
}

temeraments = ['Strict','Low','Moderate','Creative','Inventive','Crazy']

default_req_params_paraphrase = {}

default_req_params_paraphrase['Strict'] = {
    'temperature': 0.7,
    'top_p': 0.1,
    'top_k': 40,
    'repetition_penalty': 1.18
}


default_req_params_paraphrase['Low'] = {
    'temperature': 0.1,
    'top_p': 1.0,
    'top_k': 0,
    'repetition_penalty': 1.2,
}

"""default_req_params_paraphrase['Determined2'] = {
    'temperature': 0.3,
    'top_p': 0.75,
    'top_k': 40,
    'repetition_penalty': 1.2,
}
"""
default_req_params_paraphrase['Moderate'] = {
    'temperature': 0.7,
    'top_p': 0.5,
    'top_k': 40,
    'repetition_penalty': 1.2,
}

default_req_params_paraphrase['Creative'] = {
    'temperature': 1.0,
    'top_p': 0.4,
    'top_k': 40,
    'repetition_penalty': 1.2,
}

default_req_params_paraphrase['Inventive'] = {
    'temperature': 1.1,
    'top_p': 0.75,
    'top_k': 40,
    'repetition_penalty': 1.2,
}
default_req_params_paraphrase['Crazy']  = {
    'temperature': 1.75,
    'top_p': 0.6,
    'top_k': 100,
    'repetition_penalty': 1.2,
}

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_class):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_classes=elem_class)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button

def atoi(text):
    return int(text) if text.isdigit() else text.lower()

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def get_available_templates():
    paths = (x for x in Path('extensions/Playground/Paraphrase').iterdir() if x.suffix in ('.txt'))
    return ['None'] + sorted(set((k.stem for k in paths)), key=natural_keys)

def get_file_path(folder, filename):
    basepath = "extensions/Playground/"+folder
    #print(f"Basepath: {basepath} and {filename}")
    paths = (x for x in Path(basepath).iterdir() if x.suffix in ('.txt'))
    for path in paths:
        if path.stem.lower() == filename.lower():
            return str(path)
    return ""

def read_file_to_string(file_path):
    data = ''
    try:
        with open(file_path, 'r') as file:
            data = file.read()
    except FileNotFoundError:
        data = ''

    return data

def load_Paraphrase_template(file):
    global params
    template = 'Paraphrase the following\n<|context|>'
    path = get_file_path('Paraphrase',file)
    
    if path:
        print(f"Loading Paraphrase Template: {path}")
        template = read_file_to_string(path)

    params['paraph_templ_sel'] = file
    params['paraph_templ_text'] = template
    return template

def get_last_line(string):
    lines = string.splitlines()
    if lines:
        last_line = lines[-1]
        return last_line
    else:
        return ""


def generate_prompt(string,summary):

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

    memory = ''
    if params['selectedMEM']=='Memory A':
        memory = params['memoryA']  
    elif params['selectedMEM']=='Memory B':
        memory = params['memoryB'] 
    elif params['selectedMEM']=='Memory C':
        memory = params['memoryC'] 

    if params['selectedSUM']!='None':
        if memory:
            promptSUM = params['summary_include_turn2'].replace('\\n', '\n')
            promptSUM = promptSUM.replace('<|summary|>', summary)
            promptSUM = promptSUM.replace('<|memory|>', memory)
        else:
            promptSUM = params['summary_include_turn'].replace('\\n', '\n')
            promptSUM = promptSUM.replace('<|summary|>', summary)


      
        modified_string = promptSUM+modified_string
    else:
        if memory:
            modified_string = memory+'\n'+modified_string
               

    return modified_string


# template = state['turn_template'].replace(r'\n', '\n')
#     
def output_modifier(string):
    #print(f"output_modifier: {string}") 
    return string

def copynote(string):
    return string

def formatted_outputs(reply):
 
    return reply, generate_basic_html(reply)

def generate_paraphrase(question, state,selectState,paraphrase, summary):
    global params
    global paraph_undo
    global paraph_undoSEL
    selF = params[selectState][0]
    selT = params[selectState][1]
    paraph_undo = question
    if not selF==selT:
        print(f"\033[1;32;1m\nGenerarting from selected text in {selectState} and inserting after {params[selectState]}\033[0;37;0m")
        before = question[:selF]
        current = question[selF:selT]
        after = question[selT:]
   
    else:
        current = question
        before = ""
        after = ""
        params[selectState]=[0,0]
        print(f"\033[1;31;1m\nNo selection in {selectState}, reverting to full text Generate\033[0;37;0m") 

    summary_state = state.copy()

    paraph_undoSEL[0] = params[selectState][0]
    paraph_undoSEL[1] = params[selectState][1]
    
    for key, value in default_req_params.items():
        summary_state[key] = value  # Update the value in 'summary_state' with the value from 'default_req_params

    for key, value in default_req_params_paraphrase.items():
        if params['paraph_temperament'] == key:
            print(f"Temperament: {key}")
            for key2, value2 in value.items():
                print(f"{key2}:{value2}")
                summary_state[key2] = value2  # Update the to paraphrase
            break
      

    params['paraph_templ_text'] = paraphrase

    user = params['pUSER']    
    bot = params['pBOT']

    prompt = params['paraph_templ_text']
    prompt = prompt.replace('<|context|>', current)
    prompt = prompt.replace('<|user|>', user)
    prompt = prompt.replace('<|bot|>', bot)
    prompt = prompt.replace('<|prevcontext|>', before)
    prompt = prompt.replace('<|nextcontext|>', after)
    


    #prompt = generate_prompt(prompt,summary)   

    for reply in generate_reply(prompt, summary_state, eos_token = None, stopping_strings=None, is_chat=False):
        params[selectState][1] = selF+len(reply)
        reply = before+reply+after
        yield formatted_outputs(reply)
   
def set_redo(txt):
    global paraph_redo   
    global paraph_redoSEL
    paraph_redoSEL[0] = params['selectA'][0]
    paraph_redoSEL[1] = params['selectA'][1]
    paraph_redo = txt

def paraphrase_undo():
    global params
    params['selectA'][0] = paraph_undoSEL[0]
    params['selectA'][1] = paraph_undoSEL[1]
    return paraph_undo

def paraphrase_redo():
    global params
    params['selectA'][0] = paraph_redoSEL[0]
    params['selectA'][1] = paraph_redoSEL[1]
    return paraph_redo

def generate_reply_wrapperMYSEL(question, state,selectState,summary):

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
        params[selectState] = [0,0]
        before = ""
        after = ""
        print(f"\033[1;31;1m\nNo selection in {selectState}, reverting to full text Generate\033[0;37;0m") 
        

    # if use quick prompt, add \n if none
    if params['usePR']:
        if not current.endswith("\n"):
            lastline = get_last_line(current)
            if lastline.startswith("---"):
                current+="\n"

    prompt = generate_prompt(current,summary)   
    for reply in generate_reply(prompt, state, eos_token = None, stopping_strings=None, is_chat=False):
        if hasattr(shared, 'is_seq2seq') and not shared.is_seq2seq:
            reply = current + reply
        reply = before+reply+after
        yield formatted_outputs(reply)

   

def generate_reply_wrapperMY(question, state, selectState,summary):

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

    prompt = generate_prompt(question,summary)           

    for reply in generate_reply(prompt, state, eos_token = None, stopping_strings=None, is_chat=False):
        if hasattr(shared, 'is_seq2seq') and not shared.is_seq2seq:
            reply = question + reply

        if prepend_str:
            reply = prepend_str+reply

        yield formatted_outputs(reply)

def generate_summary(inptext, state, selectState):
    global params
    params[selectState] = [0,0]
    
    summary_state = state.copy()
    
    for key, value in default_req_params.items():
        summary_state[key] = value  # Update the value in 'summary_state' with the value from 'default_req_params
        
    user = params['pUSER']    
    bot = params['pBOT']  

    prompt = params['summary_turn'].replace('\\n', '\n')
    prompt = prompt.replace('<|context|>', inptext)
    prompt = prompt.replace('<|user|>', user)
    prompt = prompt.replace('<|bot|>', bot)

    for reply in generate_reply(prompt, summary_state, eos_token = None, stopping_strings=None, is_chat=False):
        yield formatted_outputs(reply)

def get_available_LORA():

    #print (f"Scaling {shared.model.base_model.scaling}")

    prior_set = ['None']
    if hasattr(shared.model,'peft_config'):
        for adapter_name in shared.model.peft_config.items():
            print(f"Found adapter: {adapter_name[0]}")
            prior_set.append(adapter_name[0])

    return prior_set      

def set_LORA(item):
      
    prior_set = list(shared.lora_names)
    if hasattr(shared.model, 'set_adapter') and hasattr(shared.model, 'active_adapter'):
        if prior_set:
            if item =='None' and hasattr(shared.model.base_model, 'disable_adapter_layers'):
                shared.model.base_model.disable_adapter_layers()
                print (f"[Disable] Adapter layers for: {shared.model.active_adapter}")   
            else:
                shared.model.set_adapter(item)
                print (f"Set active adapter: {shared.model.active_adapter}")  
                if hasattr(shared.model.base_model, 'enable_adapter_layers'):
                    shared.model.base_model.enable_adapter_layers()
                    print (f"[Enable] Adapter layers")  
                
            
def get_available_loras():
    return sorted([item.name for item in list(Path(shared.args.lora_dir).glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=natural_keys)

def ui():
    #input_elements = list_interface_input_elements(chat=False)
    #interface_state = gr.State({k: None for k in input_elements})

    global params
    global selected_lora
    model_name = getattr(shared.model,'active_adapter','None')

    for key in defaultTemp_keys:
        params[key] = defaultTemp[key]

    try:
        with open(file_nameJSON, 'r') as json_file:
            new_params = json.load(json_file)
            for item in new_params:
                params[item] = new_params[item]
    except FileNotFoundError:
        params['max_words'] = 0

    params['selectA'] = [0,0]
    params['selectB'] = [0,0]
    params['selectedMEM']='None'

    if params['paraph_templ_text']:
        paraphrase_text = params['paraph_templ_text']
    else:
        paraphrase_text = load_Paraphrase_template(params['paraph_templ_sel'])
        params['paraph_templ_text'] = paraphrase_text


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
                                    toNoteB = ToolButton(value=right_symbol)                             
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
                with gr.Tab('LoRA-Rama'):
                    with gr.Column():
                        with gr.Row():
                            loramenu = gr.Dropdown(multiselect=False, choices=get_available_loras(), value=shared.lora_names, label='LoRA and checkpoints', elem_classes='slim-dropdown')
                            create_refresh_button(loramenu, lambda: None, lambda: {'choices': get_available_loras(), 'value': shared.lora_names}, 'refresh-button')
                        with gr.Row():                            
                            lorasub = gr.Radio(choices='', value=shared.lora_names, label='Checkpoints')
                            
                        with gr.Row():
                            displaytext = gr.Markdown(value='')
                            lora_apply = gr.Button(value='Apply', elem_classes='small-button')
                with gr.Tab('Perma-Memory'):
                    with gr.Column():
                        text_MEMA = gr.Textbox(value=params['memoryA'], lines=5, label='Memory A')
                        text_MEMB = gr.Textbox(value=params['memoryB'], lines=5, label='Memory B')
                        text_MEMC = gr.Textbox(value=params['memoryC'], lines=5, label='Memory C')
                        
                        with gr.Row():
                            gr.Markdown('Text flow is: Summary + Perma-Memory + Limit_Memory ( previous context )') 
                            save_btn = gr.Button('Save', elem_classes="small-button")
                with gr.Tab('Summary'):
                    with gr.Column():
                        summary_turn = gr.Textbox(value=params['summary_turn'], lines=1, label='Summarize Template Instruct: <|user|> <|bot|> <|context|>')
                        text_Summary = gr.Textbox(value=params['text_Summary'], lines=10, label='Summary:')
                        
                        sumarize_btn = gr.Button('Summarize text in Notebook A', variant='primary')
                        summary_include_turn = gr.Textbox(value=params['summary_include_turn'], lines=1, label='In Context Summary Template: <|summary|>')
                        summary_include_turn2 = gr.Textbox(value=params['summary_include_turn2'], lines=1, label='In Context Summary + Memory Template: <|summary|> <|memory|>')
                        with gr.Row():
                            reset_def_btn = gr.Button('Templates Reset', elem_classes="small-button")
                            save_btn2 = gr.Button('Save', elem_classes="small-button")
                with gr.Tab('Paraphrase'):
                    with gr.Column():
                        with gr.Row():
                            para_templates_drop  = gr.Dropdown(choices=get_available_templates(), label='Paraphrase Instruction', elem_id='character-menu', info='Template to invoke paraphrtasing.', value=params['paraph_templ_sel'])
                            create_refresh_button(para_templates_drop, lambda: None, lambda: {'choices': get_available_templates()}, 'refresh-button')
                        with gr.Accordion(label = "Edit Template", open=True):
                            para_template_text = gr.Textbox(value=paraphrase_text, lines=10, label='Template', interactive=True)
                        with gr.Row():
                            gr_temperament = gr.Radio(choices=temeraments, value=params['paraph_temperament'], label='Temperament', interactive=True)    
                        with gr.Row():
                            paraphrase_btn = gr.Button('Rewrite [SEL]', variant='primary',elem_classes="small-button")
                            paraphrase_btn2 = gr.Button('Try Again', elem_classes="small-button")
                            paraphrase_btn_undo = gr.Button('Undo',elem_classes="small-button")
                            paraphrase_btn_redo = gr.Button('Redo',elem_classes="small-button")
                            save_btn3 = gr.Button('Save', elem_classes="small-button")
                            
                        with gr.Row():
                            gr.Markdown('Select some text in Notebook A and press Paraphrase')  
 
            with gr.Row():
                with gr.Box():
                    with gr.Column():
                        gr_summarymenu = gr.Radio(choices=['None','Summary'], value='None', label='Insert Summary', interactive=True)
                        gr_memorymenu = gr.Radio(choices=['None','Memory A','Memory B','Memory C'], value='None', label='Insert Perma-Memory', interactive=True)
                        with gr.Row():
                            max_words = gr.Number(label='Limit previous context to last # of words (0 is no limit, 500 is about half page)', value=params['max_words'])                            
                        with gr.Row():
                            gr_Loralmenu = gr.Radio(choices=get_available_LORA(), value=model_name, label='Activate Loaded LORA adapters', interactive=True)
                            create_refresh_button(gr_Loralmenu, lambda: None, lambda: {'choices': get_available_LORA(),'value': getattr(shared.model, 'active_adapter', None)}, 'refresh-button')      
                        with gr.Row():                            
                            gr.Markdown('v 6.20 by FPHam https://github.com/FartyPants/Playground')    


    selectStateA = gr.State('selectA')
    selectStateB = gr.State('selectB')
        #shared.gradio['Undo'] = gr.Button('Undo', elem_classes="small-button")
        #shared.gradio['Regenerate'] = gr.Button('Regenerate', elem_classes="small-button")
    # Todo:
    # add silider for independend temperature, top_p and top_k
    # shared.input_elements, shared.gradio['top_p'] 
    input_paramsA = [text_boxA,shared.gradio['interface_state'],selectStateA,text_Summary]
    output_paramsA =[text_boxA, htmlA]
    input_paramsB = [text_boxB,shared.gradio['interface_state'],selectStateB,text_Summary]
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
        try:
            with open(file_nameJSON, 'w') as json_file:
                json.dump(params, json_file,indent=2)
                print(f"Saved: {file_nameJSON}")
        except IOError as e:
            print(f"An error occurred while saving the file: {e}")  
    
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
                
    def update_memorymenu(x,y):
        global params
        params.update({"selectedMEM": x})
        params.update({"selectedSUM": y})

    def update_summary_turn(x,y,z):
        global params
        params.update({"summary_turn": x})
        params.update({"summary_include_turn": y})
        params.update({"summary_include_turn2": z})

    def update_summary_text(x):
        global params
        params.update({"text_Summary": x})
    
    def update_temperament(x):
        global params
        params.update({"paraph_temperament": x})
    
    def update_paraph_template(x):
        global params
        params.update({"paraph_templ_text": x})
        save_pickle()         
    
    def reset_defaults():
        global params 

        for key in defaultTemp_keys:
            params[key] = defaultTemp[key]   
        save_pickle()
        return params['summary_include_turn'],params['summary_include_turn2'],params['summary_turn']

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

    #sumarize
    summary_turn.change(update_summary_turn,[summary_turn,summary_include_turn,summary_include_turn2],None)
    summary_include_turn.change(update_summary_turn,[summary_turn, summary_include_turn,summary_include_turn2],None)
    summary_include_turn2.change(update_summary_turn,[summary_turn, summary_include_turn,summary_include_turn2],None)
    reset_def_btn.click(reset_defaults,None,[summary_include_turn,summary_include_turn2,summary_turn])

    text_Summary.change(update_summary_text,text_Summary,None)
    input_summary = [text_boxA,shared.gradio['interface_state'],selectStateA]
    output_summary = [text_Summary,htmlB]
   
    sumarize_btn.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
          generate_summary, inputs=input_summary, outputs= output_summary, show_progress=False)

    #paraphrase
    para_templates_drop.change(load_Paraphrase_template,para_templates_drop,para_template_text).then(save_pickle,None,None) 
    input_paraphrase = [text_boxA, shared.gradio['interface_state'], selectStateA, para_template_text,text_Summary]
    output_paraphrase = [text_boxA,htmlB]
   
    paraphrase_btn.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
          generate_paraphrase, inputs=input_paraphrase, outputs= output_paraphrase, show_progress=False).then(
        set_redo,text_boxA,None)

    paraphrase_btn2.click(paraphrase_undo,None,text_boxA).then(
        gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
          generate_paraphrase, inputs=input_paraphrase, outputs= output_paraphrase, show_progress=False).then(
        set_redo,text_boxA,None)

    save_btn3.click(update_paraph_template,para_template_text,None)
    paraphrase_btn_undo.click(paraphrase_undo,None,text_boxA)
    paraphrase_btn_redo.click(paraphrase_redo,None,text_boxA)

    gr_temperament.change(update_temperament,gr_temperament,None)

    #memory
    text_MEMA.change(update_mmemory,[text_MEMA,text_MEMB,text_MEMC],None)
    text_MEMB.change(update_mmemory,[text_MEMA,text_MEMB,text_MEMC],None)
    text_MEMC.change(update_mmemory,[text_MEMA,text_MEMB,text_MEMC],None)

    gr_memorymenu.change(update_memorymenu,[gr_memorymenu,gr_summarymenu],None)
    gr_summarymenu.change(update_memorymenu,[gr_memorymenu,gr_summarymenu],None)

    usePR.change(update_activate, usePR, None)   
    text_USR.change(update_stringU, text_USR, None) 
    text_BOT.change(update_stringB, text_BOT, None) 
    preset_type.change(update_preset,preset_type,[text_USR,text_BOT])
    max_words.change(update_max_words,max_words,None)
    save_btn.click(save_pickle,None,None)
    save_btn2.click(save_pickle,None,None)
    

    gr_Loralmenu.change(set_LORA,gr_Loralmenu,None)

    def list_subfolders(directory):
        subfolders = []
        subfolders.append('None')
        for entry in os.scandir(directory):
         
            if entry.is_dir() and entry.name != 'runs':
                subfolders.append(entry.name)

        return subfolders

    def path_from_selected(selectlora,selectsub):
        if selectsub and selectsub!='None':
            return f"{selectlora}/{selectsub}"
        
        return f"{selectlora}"

   
    def apply_lora(selectlora,selectsub):
        global selected_lora
        path = path_from_selected(selectlora,selectsub)   
        selected_lora = path
        yield (f"Applying the following LoRAs to {shared.model_name} : {selected_lora}"),(f"Applying the following LoRA to {shared.model_name} : {selected_lora}")
        add_lora_to_model([selected_lora])
        if len(shared.lora_names)>0:
            yield ("Successfuly applied the LoRA"),("Successfuly applied the LoRA")   
        else:
            yield ("Lora failed..."),("LoRA failed")       

    def update_lotra_subs(selectlora):

        if selectlora:
            model_dir = f"{shared.args.lora_dir}/{selectlora}"  # Update with the appropriate directory path
            subfolders = list_subfolders(model_dir)
            return gr.Radio.update(choices=subfolders,value='None') 

        return gr.Radio.update(choices=[],value='')    

    def update_activeAdapters():
        return gr.Radio.update(choices=get_available_LORA(), value= getattr(shared.model, 'active_adapter', None))

    #loramenu, lambda: None, lambda: {'choices': get_available_loras(), 'value': shared.lora_namesgr_Loralmenu
    loramenu.change(update_lotra_subs,loramenu, lorasub)
    lora_apply.click(apply_lora,[loramenu, lorasub],[displaytext,shared.gradio['model_status']]).then(lambda : selected_lora,None, shared.gradio['lora_menu']).then(
        update_activeAdapters,None, gr_Loralmenu)
    
    #lorasub.change(path_from_selected,[loramenu,lorasub],displaytext)

 



