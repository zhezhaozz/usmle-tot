import re
import json

def escape_inner_quotes(text):
    return re.sub(r'"', r'\"', text)

def fix_json_format_response(response: str):
    #print(response)
    if "}" not in response:
        response += "}"
    if "\"\"" in response:
        response = response.replace("\"\"", "\"")
    if response.count("\"") %2 !=0 and response.rfind("\"}") == -1 :
        response = response.replace("}", "\"}")
    if ")}"in response:
        response=response.replace(")}", "}")
    if ".}"in response:
        response=response.replace(".}", "}")
    if "\n"in response:
        response=response.replace("\n", "")
    
    match = re.search(r'"plan"\s*:\s*"(.+)"', response)
    if match:
        value = match.group(1)
        fixed_value = escape_inner_quotes(value)
        response = response.replace(value, fixed_value)

    match = re.search(r'"reasoning"\s*:\s*"(.+)"', response)
    if match:
        value = match.group(1)
        fixed_value = escape_inner_quotes(value)
        response = response.replace(value, fixed_value)

    match = re.search(r'"choice"\s*:\s*"(.+)"', response)
    if match:
        value = match.group(1)
        fixed_value = escape_inner_quotes(value)
        response = response.replace(value, fixed_value)

    match = re.search(r'"answer"\s*:\s*"(.+)"', response)
    if match:
        value = match.group(1)
        fixed_value = escape_inner_quotes(value)
        response = response.replace(value, fixed_value)

    match = re.search(r'"ddx"\s*:\s*"(.+)"', response)
    if match:
        value = match.group(1)
        fixed_value = escape_inner_quotes(value)
        response = response.replace(value, fixed_value)

    # the action of eliminating intext quotes may generate extra \
    if "\\\\" in response:
        response = response.replace("\\\\", "\\")
    if "\\-" in response:
        response = response.replace("\\-", " ")
    if ".\\" in response:
        response = response.replace(".\\", ". ")
        
    return response

def clean_json_response(response, mode="generation"):
    if "}" not in response:
        response += "}"
    match = re.findall(r'\{.*?\}', response, re.DOTALL)
    if match:
        if len(match) > 1:
            match = [match[-1]]
        json_str = fix_json_format_response(match[-1].strip())  # Extract matched JSON string
        #print(json_str)
        try:
            # Try to load it as a JSON object to ensure it's valid
            json_data = json.loads(json_str)
            if mode in ["tot_termination", "generation"]:
                return json_data['answer']
            elif mode == "tot_evaluation":
                choice = json_data['choice']
                choice = int(choice)
                return choice
            elif mode == "tot_plan_generation":
                return json_data['plan']
            elif mode == "tot_reasoning_generation":
                return json_data['reasoning']
        except:
            print(f"Error decoding JSON:")
            #print(response)
            print(match[-1])
            print(json_str)
            return None
    else:
        print("No JSON object found in the input string.")
        print(response)
        return None

def parse_json_output(content, mode="generation"):
    content = fix_json_format_response(content)
    match = re.findall(r'\{.*?\}', content, re.DOTALL)
    if match:
        return clean_json_response(content, mode)
    else:
        print("Cannot decode json output")
        print(content)
        return None




