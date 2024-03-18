import argparse
import os

from config import DefaultLLM, DefaultEmbedder, get_driver
from lavague.ActionEngine import ActionEngine, MAX_CHARS
from tqdm import tqdm
import re

def extract_first_python_code(markdown_text):
    # Pattern to match the first ```python ``` code block
    pattern = r"```python(.*?)```"
    
    # Using re.DOTALL to make '.' match also newlines
    match = re.search(pattern, markdown_text, re.DOTALL)
    if match:
        # Return the first matched group, which is the code inside the ```python ```
        return match.group(1).strip()
    else:
        # Return None if no match is found
        return None

def process_instruction(query, driver, action_engine):
    
    state = driver.page_source
    query_engine = action_engine.get_query_engine(state)
    response = query_engine.query(query)
    source_nodes = response.get_formatted_sources(MAX_CHARS)
    return response.response, source_nodes

def process_file(file_path):
    
    import inspect

    # Gets the original source code of the get_driver method
    source_code = inspect.getsource(get_driver)
    
    # Split the source code into lines and remove the first line (method definition)
    source_code_lines = source_code.splitlines()[1:]
    source_code_lines = [line.strip() for line in source_code_lines[:-1]]

    # Execute the import lines
    import_lines = [line for line in source_code_lines if line.startswith("from") or line.startswith("import")] 
    exec("\n".join(import_lines))
    # Join the lines back together
    output = "\n".join(source_code_lines)
    
    driver = get_driver()
    llm = DefaultLLM()
    embedder = DefaultEmbedder()
    action_engine = ActionEngine(llm, embedder)
    
    with open(file_path, "r") as file:
        instructions = file.readlines()
    
    base_url = instructions[0]
    instructions = instructions[1:]

    driver.get(base_url)
    output += f"\ndriver.get('{base_url.strip()}')\n"
    
    template_code = """\n\n########################################\n\n# Query: {instruction}\n\n# Code: \n\n{code}"""
    
    for instruction in tqdm(instructions):
        print(f"Processing instruction: {instruction}")
        code, source_nodes = process_instruction(instruction, driver, action_engine)
        code = extract_first_python_code(code)
        exec(code)
        
        output += template_code.format(instruction=instruction, code=code).strip()
    
    output_fn = file_path.split(".")[0] + ".py"
    print(f"Saving output to {output_fn}")
    with open(output_fn, "w") as file:
        file.write(output)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Process a file.')

    # Add the arguments
    parser.add_argument('file_path',
                        metavar='file_path',
                        type=str,
                        help='the path to the file')

    # Execute the parse_args() method
    args = parser.parse_args()

    # Now you can use args.file_path
    process_file(args.file_path)

