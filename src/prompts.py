io_prompt = {
    "system_message": "You are a medical student taking the UMSLE exam. Answer the following multiple-choice medical question.",
    "user_message": """Question:
                    {question}

                    Options:
                    {options}

                    Respond in this EXACT JSON format: {\"answer\": <provide the letter of the option you choose, e.g., A, B, C, or D>}. PLEASE MAKE SURE YOUR OUPUT FOLLOWS THE JSON FORMAT STRICTLY"""
}