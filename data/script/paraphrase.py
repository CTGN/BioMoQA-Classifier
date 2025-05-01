import pandas as pd
from datasets import Dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch

def paraphrase_dataset(
    training_df,
    batch_size=16,
    similarity_threshold=0.85,
    device=0 if torch.cuda.is_available() else -1
):
    hf_dataset = Dataset.from_pandas(training_df)

    # Load models with GPU optimization
    translate_pipeline = pipeline(
        "translation",
        model="facebook/nllb-200-3.3B",
        src_lang="eng_Latn",
        tgt_lang="fra_Latn",
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32
    )

    similarity_model = SentenceTransformer(
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        device='cuda' if device == 0 else 'cpu'
    )

    def batch_back_translate(batch):
        # Forward translation (English -> French)
        fr_translations = translate_pipeline(
            batch["text"],
            max_length=1024,
            batch_size=batch_size,
            clean_up_tokenization_spaces=True
        )

        # Back translation (French -> English)
        en_back_translations = translate_pipeline(
            [x["translation_text"] for x in fr_translations],
            src_lang="fra_Latn",
            tgt_lang="eng_Latn",
            max_length=1024,
            batch_size=batch_size,
            clean_up_tokenization_spaces=True
        )

        return {"text": [x["translation_text"] for x in en_back_translations]}

    def validate_semantics(batch):
        # Encode original and paraphrased texts
        orig_embeddings = similarity_model.encode(
            batch["abstract_original"],
            batch_size=batch_size,
            convert_to_tensor=True
        )
        
        para_embeddings = similarity_model.encode(
            batch["text"],
            batch_size=batch_size,
            convert_to_tensor=True
        )

        # Compute cosine similarities
        similarities = torch.nn.functional.cosine_similarity(
            orig_embeddings, para_embeddings
        )

        # Filter based on similarity threshold
        return {
            "text": [text if sim >= similarity_threshold else orig
                        for text, orig, sim in zip(
                            batch["text"],
                            batch["abstract_original"],
                            similarities
                        )],
            "labels": batch["labels"]
        }

    # Step 1: Generate paraphrases
    para_dataset = hf_dataset.map(
        batch_back_translate,
        batched=True,
        batch_size=batch_size,
        remove_columns=hf_dataset.column_names
    )

    # Add original texts for validation
    para_dataset = para_dataset.add_column("abstract_original", hf_dataset["text"])

    # Step 2: Semantic validation
    para_dataset = para_dataset.map(
        validate_semantics,
        batched=True,
        batch_size=batch_size,
        remove_columns=["abstract_original"]
    )

    return para_dataset.to_pandas()

df = pd.read_csv('data/train/train_pubmed_positive.csv')
df_praph = paraphrase_dataset(df)
df_praph.to_csv("data/train/paraphrased.csv")
print("Before paraphrasing :",df.iloc[0])
print("After paraphrasing :",df_praph.iloc[0])