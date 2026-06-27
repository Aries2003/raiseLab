import pandas as pd
import ollama 
import os

DATA_PATH = r"dataset\XSUM\validation-00000-of-00001 (1).parquet" 
OUTPUT_PATH = r"output\Summarization\llama_summary.parquet" 

MODEL_NAME = "llama3.2:1b" 

def prompt_builder(row):
    article_text = row["document"]
    summ = row["summary"]
    id = row["id"]


    prompt = (
        f"Context Article: \n {article_text}\n\n"
        f"Provide a single sentence, highly concise summary of the article above."
    )

    return prompt 

def query_model(formatted_prompt ):

    system_instruction = ("""You are expert journalist tasked with extreme summurization. \n
                            Read the provided article and summarize it in exactly ONE concise sentence\n.
                            Do not include introductory phrasing like 'Here is a summary:' or 'This article discusses'.
                            Output only the final summary sentence directly""")

    response = ollama.chat(
        model= MODEL_NAME,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": formatted_prompt}
            
        ],
        options={
            "temperature": 0.0
            }
    )
     
    return response['message']['content'].strip()

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset doesnt exist")

        return
    
    df = pd.read_parquet(DATA_PATH)
    print(f"Dataset loaded {len(df)}")

    model_summaries = []

    for index, row in df.iterrows():

        prompt = prompt_builder(row)

        if index % 50 ==0:
            print(f"Evaluating instance {index}/ {len(df)}.....")

        try: 
            generatd_summaries = query_model(prompt)
            model_summaries.append(generatd_summaries)
        except Exception as e:
            print(f"ERROR at indexx {index}: {e}")
            model_summaries.append("ERROR: Generation Failed")

    
    df['model_summary'] = model_summaries
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Evaluation Complete! Results at {OUTPUT_PATH}")

def check_output():
    
    df = pd.read_parquet(OUTPUT_PATH)
    index =19
    row = df.iloc[index]
    print(f'\n Article: {row["document"]}\n')
    print(f'Summary: {row["summary"]}\n')
    print(f'Model Generated: {row['model_summary']}')
    # # print(df.iloc[index, 1:4])

def single_query_run():
    df = pd.read_parquet(DATA_PATH)
    index =19
    p = prompt_builder(df.iloc[index])
    try:
        generated_summaries = query_model(p)
        
        print(f"Article:\n {df.document[index]}\n")
        print(f"Summary: {df.summary[index]}\n")
        print(f"Model Generated: {generated_summaries}")
    except Exception as e:
        print(f"Error at index : {e}")

if __name__ == "__main__":
    # main()
    # single_query_run()
    check_output()