import pandas as pd
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

def back_translate(text: str, translator: pipeline) -> str:
    """
    Performs back-translation using FLAN-T5 model with explicit translation prompts.
    
    Args:
        text (str): Input text to back-translate
        translator: Initialized text2text-generation pipeline
    
    Returns:
        str: Back-translated text (or original if error occurs)
    """
    try:
        # English -> French
        fr_prompt = f"Translate English to French: {text}"
        french = translator(fr_prompt, max_new_tokens=512)[0]['generated_text']
        
        # French -> English
        en_prompt = f"Translate French to English: {french}"
        english = translator(en_prompt, max_new_tokens=512)[0]['generated_text']
        
        return english
    except Exception as e:
        print(f"Back-translation error: {e}")
        return text  # Fallback to original text

def back_translation(csv_path: str) -> pd.DataFrame:
    """
    Main function to process CSV and generate back-translated dataset
    
    Args:
        csv_path (str): Path to input CSV file with 'abstract' and 'label' columns
    
    Returns:
        pd.DataFrame: New DataFrame with back-translated abstracts and original labels
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Initialize FLAN-T5 pipeline
    translator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        tokenizer=T5Tokenizer.from_pretrained("google/flan-t5-small"),
        model_kwargs={"temperature": 0},
        device=0 
    )
    
    # Process all abstracts
    df['back_translated'] = df['abstract'].progress_apply(
        lambda x: back_translate(x, translator)
    )
    
    return pd.DataFrame({
        'abstract': df['back_translated'],
        'label': df['label']
    })


df = pd.read_csv('data/train/train_pubmed_positive.csv')
df_backtrans = back_translation(df)
df_backtrans.to_csv("data/train/back_trans.csv")
print("Before back translation :",df.iloc[0])
print("After back translation :",df_backtrans.iloc[0])