from openai import AzureOpenAI, OpenAI
import os
import sys
import fitz 
import yaml 

#################################code for blending line with abstract##################################

def extract_abstract(text):
    abstract_start = "abstract"
    abstract_end = "\n1\n"  # Assuming the abstract ends with a double newline

    start_index = text.lower().find(abstract_start)
    end_index = text.find(abstract_end, start_index)
    
    if start_index != -1 and end_index != -1:
        abstract = text[start_index+8:end_index].strip()
    else:
        abstract = "Abstract not found or could not be extracted."
    return abstract, start_index+8

def blend_abstract(prompt, wm, abstract, client, model):
    prequel = prompt['before'] + wm + prompt['after']
    instruction = 'Blend the following sentence naturally into the following paragraph. Maintain the sentence exactly as is.' + 'Begin Sentence:' + prequel + 'End Sentence' + 'Begin Paragraph:' + abstract + 'End Paragraph'

    messages = [ {"role": "user", "content": instruction }]
    try:
        response = client.chat.completions.create(model=model_name, messages=messages)
        return response.choices[0].message.content
    except (KeyError, IndexError, TypeError) as e:
        return f"[Error parsing response: {e}]"

def add_abstract_blend_wm(file_name, client, model, cfg):
    content = extract_text(file_name)
    text = content['text']
    abstract, start_pos = extract_abstract(text)
    blended_abstract = blend_abstract(cfg['prompt'], cfg['wm'], abstract, client, model)

    content['text'] = text[:start_pos] + blended_abstract + text[start_pos + len(abstract):]
    return content

#################################################################################################


def truncate_by_estimated_tokens(text, max_tokens=1000):
    estimated_words = int(max_tokens * 0.75)
    return ' '.join(text.split()[:estimated_words])

def build_reviewer_message(reviewer_prompt, file_content, cfg):
    #if there is a token limit, then truncate the text of the file approximately
    if 'max_tokens' in cfg:
        file_content['text'] = truncate_by_estimated_tokens(file_content['text'], cfg['max_tokens'])
        
    prompt_text = reviewer_prompt['before'] + file_content['text'] + reviewer_prompt['after']        
    return [{"role": "user", "content": prompt_text}]

def extract_text(file_path):
    #given an open pdf file descriptor return text from it
    doc = fitz.open(file_path)
    all_text = [page.get_text() for page in doc if page.get_text().strip()]
    doc.close()
    return {'text': "\n".join(all_text), 'image_paths': []}

def add_wm(input_pdf_name, cfg):
    doc = fitz.open(input_pdf_name)
    
    page_number = max(0, cfg['page_number'])
    if page_number >= len(doc):
        raise ValueError("Invalid page number")
    
    text = cfg['prompt']['before'] + cfg['wm'] + cfg['prompt']['after']

    page = doc[page_number]
    page.insert_text(cfg['position'], text, fontsize = cfg['fontsize'], color= cfg['color'])
    output_fname = input_pdf_name[:-4] + "_wm.pdf"
    doc.save(output_fname)
    doc.close()

def test(pdf_file_name, client, model_name, extract_content, test_cfg):
    #given an open pdf_file descriptor, extract content from the file
    #then send it to the model with added reviewer prompt, and return text of the response
    content = extract_content(pdf_file_name)
    return test_content(content, client, model_name, test_cfg)
    

def test_content(content, client, model_name, test_cfg):
    expt_cfg = {k: v for k, v in test_cfg.items() if k != 'reviewer_prompt'}
    messages = build_reviewer_message(test_cfg['reviewer_prompt'], content, expt_cfg)

    try:
        response = client.chat.completions.create(model=model_name, messages=messages)
        return response.choices[0].message.content
    except (KeyError, IndexError, TypeError) as e:
        return f"[Error parsing response: {e}]"
        
def filter_by_presence(review, watermark): 
    return watermark in review

def save_results(file_name, model_name, results, config_wm, config_test):
    experiment_data = { 'config_wm': config_wm, 'config_test': config_test, 'model': model_name, 'result': results }
    with open(file_name, 'w') as file:
        yaml.dump(experiment_data, file, default_flow_style=False)

