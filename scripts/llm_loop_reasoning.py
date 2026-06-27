
import os
import pandas as pd
import ollama




DATA_PATH = r"dataset\NLG\validation-00000-of-00001.parquet"
OUTPUT_PATH= r"output\Reasoning\llama_reasoning.parquet"

MODEL_NAME = "llama3.2:1b" 


def prompt_builder(row):
    obs1 = row["observation_1"]
    obs2 = row["observation_2"]
    h1 = row["hypothesis_1"]
    h2 = row["hypothesis_2"]

    prompt = (
        f"A narrative timeline has a beggining and an end , but is missing the middle event \n\n"
        f"Beginning (Observation 1): {obs1} \n"
        f"Ending (Observation 2): {obs2} \n\n"
        f"Which of the following hypothesis is the most plausible middle event that connects them \n"
        f"1) {h1}\n"
        f"2) {h2}\n"
    )
    # print(prompt)
    return prompt,h1,h2

def query_model(formatted_prompt):

    system_instruction = """You are an expert in logic and commonsense reasoning . \n
                          Analyze the provided observation and select the most plausible hypothesis.\n 
                          Respond with EXACTLY the digit '1' or '2' corresponding to the correct hypothesis \n
                          Do NOT write any introduction, punctuation, or explainations. Output only a single number character """ 
    
    response = ollama.chat(
        model = MODEL_NAME,
        messages= [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": formatted_prompt}
        ],
        options={
            "temperature": 0.0,
            # "num_predict":5
        }

    )

    return response['message']['content'].strip()

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return
    
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} dataset instance. Starting Evaluation")

    model_answers = []

    for index, row in df.iterrows():

        prompt, h1,h2 = prompt_builder(row)

        if index % 50 == 0:

            print(f"Evaluating instance {index}/{len(df)}...")

        try:
            ans= query_model(prompt)

            if '1' in ans:
                model_answers.append(h1)
            elif '2' in ans:
                model_answers.append(h2)
            else:
                model_answers.append(-1)
        except Exception as e:
            print(f"Error at index {index}: {e}")
            model_answers.append(-1)

    df['model_answer'] = model_answers
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Evaluation complete! Results exported to {OUTPUT_PATH}")

def check_output():
    
    df = pd.read_parquet(OUTPUT_PATH)
    index =19
    row = df.iloc[index]
    print(f'\nObservation 1: {row["observation_1"]}')
    print(f'Observation 2: {row["observation_2"]}')
    print(f'\n Hypo 1: {row["hypothesis_1"]}')
    print(f'Hypo 2: {row["hypothesis_2"]}')
    print(f'Answer: {row["label"]}')
    print(f'Model selected: {row["model_answer"]}')
    # print(df.iloc[index, 1:4])

def single_query_run():

    index = 11
    df = pd.read_parquet(DATA_PATH)
    p, hypothesis_1, hypothesis_2 = prompt_builder(df.iloc[index])
    
    try:
        ans = query_model(p)
        print(p)

        if '1' in ans:
            print(f"Model chose 1: {hypothesis_1}")
        elif '2' in ans:
            print(f"Model chose 2: {hypothesis_2}")
        else:
            print(f"Malformed model response: {ans}")
    except Exception as e:
        print(f"Error at index {index}: {e}")

if __name__ == "__main__":
    # main()
    single_query_run()
    # check_output()

    