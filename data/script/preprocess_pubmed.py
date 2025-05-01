from preprocess_ipbes import *

def optimize_df_memory(df):
    """Optimize DataFrame memory usage by downcasting numeric columns and converting objects to categories."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'object':
            if df[col].nunique() < len(df[col]) * 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
    return df

def clean_pubmed(df):
    """
    Deletes duplicates and conflicts in the dataset.
    """
    df = optimize_df_memory(df)
    print(df.head())

    print(df.isnull().sum())

    X = df["text"]
    y = df["labels"]

    duplicates = df[df.duplicated(subset=['text'], keep=False)]
    print(f"Total duplicate abstracts: {len(duplicates)}")
    print(duplicates[['text', 'labels','domain']].sort_values('text'))

    df_clean = df.drop_duplicates(subset=['text'], keep='first')

    conflicts = df.groupby('text')['labels'].nunique().reset_index()
    conflicts = conflicts[conflicts['labels'] > 1]

    print(f"Abstracts appearing in both classes: {len(conflicts)}")
    print(conflicts)

    # Free memory
    del duplicates
    del conflicts
    gc.collect()
    return df_clean


def pmids_to_text(train_df, test_df):
    """Write PMIDs to files in chunks to reduce memory usage."""
    chunk_size = 10000
    
    for df, prefix in [(train_df, 'train'), (test_df, 'test')]:
        for label, label_text in [(1, 'pos'), (0, 'neg')]:
            pmids = (df[df["labels"]==label]['PMID']).dropna().astype(int).astype(str)
            
            with open(f"{prefix}_{label_text}_pmids.txt", "w") as f:
                for i in range(0, len(pmids), chunk_size):
                    chunk = pmids[i:i + chunk_size].tolist()
                    f.write(" ".join(chunk) + " ")
            
            print(f"\nSaved {len(pmids)} PMIDs to {prefix}_{label_text}_pmids.txt")
            del pmids
            gc.collect()

def preprocess_pubmed(df):
    df = optimize_df_memory(df)
    #Whole process of cleaning and splitting the data
    #We first clean the data
    df_clean = clean_pubmed(df)
    train_df, dev_df, test_df = stratified_split(df_clean, target_column='labels')


    print(f"Train size: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"Dev size: {len(dev_df)} ({len(dev_df)/len(df):.1%})")
    print(f"Test size: {len(test_df)} ({len(test_df)/len(df):.1%})")

    
    train_df[train_df['domain']=='pos'].to_csv("data/train/train_pubmed_positive.csv",index=False)
    train_df[train_df['domain']=='in-neg'].to_csv("data/train/train_pubmed_negative.csv",index=False)
    train_df[train_df['domain']=='out-neg'].to_csv("data/train/train_arxiv_negative.csv",index=False)

    # Write files in chunks
    chunk_size = 10000
    for chunk_start in range(0, len(df_clean), chunk_size):
        chunk = df_clean[chunk_start:chunk_start + chunk_size]
        mode = 'w' if chunk_start == 0 else 'a'
        chunk.to_csv("data/corpus/all.clean.csv", mode=mode, header=(mode=='w'), index=False)
    
    for chunk_start in range(0, len(train_df), chunk_size):
        chunk = train_df[chunk_start:chunk_start + chunk_size]
        mode = 'w' if chunk_start == 0 else 'a'
        chunk.to_csv("data/train/train.csv", mode=mode, header=(mode=='w'), index=False)
    
    for chunk_start in range(0, len(test_df), chunk_size):
        chunk = test_df[chunk_start:chunk_start + chunk_size]
        mode = 'w' if chunk_start == 0 else 'a'
        chunk.to_csv("data/test/test.csv", mode=mode, header=(mode=='w'), index=False)
    
    return train_df, test_df, df_clean

if __name__ == "__main__":

    print("Pubmed data preprocessing pipeline")
    df=pd.read_csv("data/corpus/all.csv")

    df=df.rename(columns={"label": "labels"})
    train_df,test_df,df_clean=preprocess_pubmed(df)

    pmids_to_text(train_df,test_df)