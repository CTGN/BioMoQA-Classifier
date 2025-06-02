from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import torch
import pandas as pd
from nlpaug.augmenter.word import SynonymAug
from time import perf_counter
from sentence_transformers import SentenceTransformer

#TODO : clean the code and make functions
#Not sure if we still need this or not


# Load data
train_df = pd.read_csv('data/train/train.csv')
positives = train_df[train_df['labels'] == 1]['text']

# Model and tokenizer setup with padding configuration
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    trust_remote_code=True,
    padding_side='left'  # Important for batched inference
)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    quantization_config=quantization_config
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

def generate_synthetic_data(label, num_samples, paraphrase=False, batch_size=8):
    synthetic_data = []
    
    # Generate all prompts upfront
    all_messages = []
    for i in range(num_samples):
        sample = positives.iloc[i % len(positives)]
        
        if paraphrase:
            prompt = f"""Task: Paraphrase the following scientific abstract:
            "{sample}"
            Output Format: (150-250 words, answer only with the paraphrased abstract)"""
        else:
            prompt = f"""Task:
                Write a scientifically rigorous and well-structured abstract of publication witch Mesh term would be "Microbial Consortia", suitable for publication in a peer-reviewed journal. The abstract should reflect the complexity and depth expected in microbial ecology, synthetic biology, and bioprocess engineering.
                Guidelines:

                    Maintain an academic, formal tone—avoid unnecessary simplifications while ensuring clarity.

                    Ensure coherence, logical flow, and conciseness while integrating key scientific concepts.

                    Incorporate precise terminology relevant to microbial consortia, metabolic interactions, ecological networks, and engineered consortia.

                    If applicable, discuss real-world applications, challenges, and future research opportunities.

                    Emulate the technical depth and writing style found in the provided example abstract.

                    Don't write any keywords

                    Follow the strucuture of the provided example abstract

                Use the following Abstract for Reference:

                {sample}

                Output Format:

                - (150-250 words, following the outlined structure, answer only by giving the requested abstract nothing more)
                - One paragraph containing only the outilined abstract 
                - Don't add any comment that is not part of the abstract"""
        messages = [
            {"role": "user", "content": "You are an AI assistant for generating machine learning training data."},
            {"role": "user", "content": prompt}
        ]
        all_messages.append(messages)
    
    print("Processing...")
    # Process in batches
    outputs = pipe(
        all_messages,
        return_full_text=False,
        max_new_tokens=512,
        batch_size=batch_size,
        pad_token_id=tokenizer.pad_token_id  # Ensure proper padding
    )
    print("processed !")
    
    # Collect responses
    for i, out in enumerate(outputs):
        try:
            response = out[0]['generated_text'].strip()
            synthetic_data.append({
                "text": response,
                "labels": label,
                "domain": "pos"
            })
        except (KeyError, IndexError) as e:
            print(f"Error processing output {i}: {e}")
    
    return pd.DataFrame(synthetic_data)

# Generate synthetic data with batch processing
start=perf_counter()
num_samples=167*3
synth_df = generate_synthetic_data(label=1, num_samples=num_samples, batch_size=16)#Try 19
stop=perf_counter()

print(f"Time spent for data generation : {stop-start} ({(stop-start)/num_samples} per sample) ")

synth_df.to_csv("data/synthetic_data.csv", index=False)

# Combine and shuffle data
dataset = pd.concat([train_df, synth_df], axis=0)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
dataset.to_csv("data/processed_train.csv")


#Implement a semantic similarity check to make sure we are not too far from our positives (Look for the SOTA method to do that : I think the model sentence-transformers/all-MiniLM-L6-v2 would be a sopution)

"""
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode()
print(embeddings.shape)

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
model.rank(,synth_df["text"].to_list())
print(similarities)"
"""