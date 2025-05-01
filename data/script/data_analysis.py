import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import re

train_df=pd.read_csv("data/train/train.csv")
val_df=pd.read_csv("data/val/val.csv")
test_df=pd.read_csv("data/test/test.csv")
synth_df=pd.read_csv("data/processed_train.csv")

#Analysis

def pubmed_analysis(data,data_type):
    # Calculate normalized class proportions
    class_distribution = data["labels"].value_counts(normalize=True).reset_index()
    class_distribution.columns = ["labels", "proportion"]
    subset_prop = data['domain'].value_counts(normalize=True).reset_index()
    subset_prop.columns = ["labels", "proportion"]

    # Plot settings
    sns.set_style('whitegrid')
    colors = sns.color_palette()[:3]
    plt.figure(figsize=(8, 5))
    # Create stacked bars
    fig, ax = plt.subplots()

    prop_out=subset_prop.iloc[0]["proportion"]
    prop_in=subset_prop.iloc[1]["proportion"]
    prop_pos=class_distribution.iloc[1]["proportion"]
    prop_neg=class_distribution.iloc[0]["proportion"]

    ax.bar([0],  prop_out, bottom=[0], color=colors[0], label="out-domain")
    ax.bar([0], prop_in, bottom=[prop_out], color=colors[1], label="in-domain")
    ax.bar([1],prop_pos, bottom=[0], color=colors[2], label="positives")


    plt.text(0, prop_out/2, f"{prop_out:.1%}", ha="center")
    plt.text(0, prop_out+prop_in/2, f"{prop_in:.1%}", ha="center")

    plt.text(0, prop_neg +0.02, f"{prop_neg:.1%}", ha="center")

    plt.text(1, prop_pos + 0.02, f"{prop_pos:.1%}", ha="center")


    ax.set_ylabel('Proportion')
    ax.legend(title='data type')
    plt.title("Normalized Class Distribution (Proportions) in " + data_type +" set")
    plt.ylim(0, 1) 
    plt.savefig('plots/Class_distribution' + data_type + '.png')
    plt.show()

    print("The size of the " + d_type + " set is : ",len(data))
    print(f"Class balance in "+ data_type + " set :\n",data['labels'].value_counts())

    #Let's analyse the distribution of words into our positive training set
    text = ' '.join(data[data['domain']=='pos']['text'].astype(str).tolist())

    text = re.sub(r'[^A-Za-z\s]', '', text)

    text = text.lower()

    stopwords = set(STOPWORDS)
    text = ' '.join(word for word in text.split() if word not in stopwords)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    plt.savefig('plots/'+data_type+'PositivesWordCloud.png')
    plt.title(" Word Cloud for positive training abstracts")
    plt.show()


for data,d_type in (train_df,'Train'),(val_df,'Val'),(test_df,'Test'),(synth_df,'Synthetic'):
    pubmed_analysis(data,d_type)