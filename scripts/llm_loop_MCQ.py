from operator import index
import os
import pandas as pd
import ollama

"""ARC_DA Dataset Format"""

DATA_PATH = r"dataset\ARC-DA\ARC-Easy\validation-00000-of-00001.parquet"
OUTPUT_PATH = r"output\QA\llama_QA.parquet"

MODEL_NAME = "llama3.2:1b" 

def prompt_builder(row):
    base_question = row["question"]
    choices_dict = row["choices"]

    labels = choices_dict.get('label', [])
    texts = choices_dict.get('text', [])

    options_text = ''

    for label, text in zip(labels, texts):
        options_text += f"{label}){text}\n"

    prompt =    f"Question: {base_question}\n\nChoices:\n{options_text}"
    # print(prompt)
    return prompt


def query_model(formatted_prompt):

    system_instruction = """You are an expert science assistant taking a multiple-choice exam.\n
        Analyze the question and choices provided. Respond with EXACTLY One 
        sentence representing the correct answer choice.\n
        Do NOT write punctuation, or extra spaces."""
    
    response = ollama.chat(
        
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": formatted_prompt}
        ],
        options={
            "temperature": 0.0,
            # "num_predict":90 
        }
    )
    # print(response)
    return response['message']['content'].strip()

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} rows. Starting evaluation via Ollama ({MODEL_NAME})...")

    answers = []
    
    for index, row in df.iterrows():
        prompt = prompt_builder(row)
        
        if index % 10 == 0:
            print(f"Processing row {index}/{len(df)}...")
            
        try:
            answer = query_model(prompt)
            answers.append(answer)
        except Exception as e:
            print(f"Error at index {index}: {e}")
            answers.append("ERROR: Ollama generation failed")

    df['model_answer'] = answers
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Done! Results successfully saved to {OUTPUT_PATH}")

def check_output():
    
    df = pd.read_parquet(OUTPUT_PATH)
    index =19
    print(df.question[index])
    print(df.model_answer[index])
    # print(df.iloc[index, 1:4])

def single_query_run():
    df = pd.read_parquet(DATA_PATH)
    index = 11
    row = df.iloc[index]
    p = prompt_builder(row)
    try:
        answer = query_model(p)
        
        print(df.question[index])

        print(answer)
    except Exception as e:
        print(f"Error at index : {e}")

if __name__ == "__main__":
    # main()
    # single_query_run()
    check_output()

    




