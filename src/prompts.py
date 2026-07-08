io_prompt = {
    "system_message": "You are a medical student taking the UMSLE exam. Answer the following multiple-choice medical question.",
    "user_message": """Question:
                    {question}

                    Options:
                    {options}

                    Respond in this EXACT JSON format: {\"answer\": <provide the letter of the option you choose, e.g., A, B, C, or D>}. PLEASE MAKE SURE YOUR OUPUT FOLLOWS THE JSON FORMAT STRICTLY"""
}

tot_generation_prompt = {
    "system_message": """
    You are a medical student taking the UMSLE exam. 
    
    Given the following multiple-choice medical question, your task is to provide an analytical plan to answer this question.
    
    Do NOT directly answer the question. Provide an analytical plan ONLY.
    """,
    "user_message": """Question:
                    {question}

                    Options:
                    {options}

                    Respond in this EXACT JSON format: {\"plan\": <your analytical plan for this question>}. PLEASE MAKE SURE YOUR OUPUT FOLLOWS THE JSON FORMAT STRICTLY"""
}

tot_reasoning_generation_prompt = {
    "system_message": """
    You are a medical student taking the UMSLE exam. 
    
    Given the following multiple-choice medical question and the analytical plan, your task is to provide a detailed reasoning to answer this question.
    """,
    "user_message": """Question:
                    {question}

                    Options:
                    {options}

                    Analytical plan:

                    {plan}

                    Respond in this EXACT JSON format: {\"reasoning\": <your reasoning for this question>}. PLEASE MAKE SURE YOUR OUPUT FOLLOWS THE JSON FORMAT STRICTLY"""
}

tot_evaluation_prompt = {
    "system_message": """
    You are a medical student taking the UMSLE exam. 
    
    Given the following {PLAN_OR_REASONING}, choose the best one that can help answer the question below:
    """,
    "user_message": """Question:
                    {question}

                    Options:
                    {options}

                    choose the best {PLAN_OR_REASONING} from the following:

                    {tot_options}

                    Respond in this EXACT JSON format: {\"choice\": <the best plan you chose>}. PLEASE MAKE SURE YOUR OUPUT FOLLOWS THE JSON FORMAT STRICTLY"""
}

tot_termination_prompt = {
    "system_message": """
    You are a medical student taking the UMSLE exam. 
    
    Given the provided reasonings, answer the following question.
    """,
    "user_message": """Question:
                    {question}

                    Options:
                    {options}

                    Reasonings:

                    {reasoning}

                    Respond in this EXACT JSON format: {\"answer\": <provide the letter of the option you choose, e.g., A, B, C, or D>}. PLEASE MAKE SURE YOUR OUPUT FOLLOWS THE JSON FORMAT STRICTLY"""
}