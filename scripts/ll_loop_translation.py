import pandas as pd
import ollama
import os

DATA_PATH = r"dataset\WMT19\validation-00000-of-00001.parquet"
OUTPUT_PATH = r"output\Machine Translation\llama_translation.parquet" 

MODEL_NAME = "llama3.2:1b" 


def prompt_builder(row):
    translation = row["translation"]

    source_lang = translation.get('de')

    target_lang = translation.get('en')

    prompt = (
        f"Translate the following text from German to English.\n\n"
        f"Text: {source_lang}\n\n"
        f"Translation:"
    )

    return prompt   


def query_model(formatted_prompt):

    system_instruction = (""" You are a professional, accurate translator.\n
                          Translate the provided text into fluent English.\n
                          Do NOT include any conversational filler, explanations, notes, or quotation marks.
                          Output ONLY the raw translated text directly.  """)

    response = ollama.chat(
        model = MODEL_NAME,
        messages=[
            {"role": "system", "content": system_instruction },
            {"role": "user", "content": formatted_prompt }
        ],
        options={
            "temperature": 0.0
        }

    
    )

    return response['message']['content'].strip()


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found")

        return
    
    df = pd.read_parquet(DATA_PATH)

    print(f"Dataset loaded {len(df)}")


    model_translations = []

    for index, row in df.iterrows():

        prompt = prompt_builder(row)

        if index % 50 ==0 :
            print(f"Evaluating instance {index}/{len(df)}........")

        try :
            generated_translation = query_model(prompt)
            model_translations.append(generated_translation)
        except Exception as e:
            print(f"ERROR: at index {index}: {e}")
            model_translations.append("ERROR: Generation failed")

    df['model_translation']  = model_translations
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Evaluation Complete! Results at {OUTPUT_PATH}")


def check_output():
    
    df = pd.read_parquet(OUTPUT_PATH)
    index =19
    row = df.iloc[index]
    print(f"German:   {row["translation"].get('de')}")
    print(f"English:   {row["translation"].get('en')}")
    print(f"Model Translation:   {row.model_translation}")
    # print(df.iloc[index, 1:4])

def single_query_run():
    df = pd.read_parquet(DATA_PATH)
    index =19
    row = df.iloc[index]
    p = prompt_builder(row)
    try:
        model_translations = query_model(p)
        
        print(f"German:     {row["translation"].get('de')}")
        print(f"English:     {row["translation"].get('en')}")
        print(f"Model Translation:     {model_translations}")
    except Exception as e:
        print(f"Error at index : {e}")

if __name__ == "__main__":
    # main()
    # single_query_run()
    check_output()