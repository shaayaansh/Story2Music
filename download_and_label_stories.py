import kagglehub
import shutil
import os
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re
import ast
import csv

def download_stories():
    """
    Pull down all stories from Kaggle dataset.
    """
    source_path = kagglehub.dataset_download('ratthachat/writing-prompts')
    destination_path = os.path.join(os.getcwd(), '1')
    shutil.move(source_path, destination_path)

def get_dataframe(split, s_o_t):
    """
    Returns a dataframe for a given file

    Args:
    split (string): train, test, or valid
    s_o_t (string): Source or Target

    Returns:
    pandas dataframe
    """
    curr_dir = os.getcwd()
    input_file = os.path.join(curr_dir, f'1/writingPrompts/{split}.wp_{s_o_t}')
    output_file = os.path.join(curr_dir, f'1/{split}_{s_o_t}.csv')

    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            data.append(line)

    df = pd.DataFrame(data, columns=['Prompt'])
    #df.to_csv(output_file, index=False)

    print(f'Converted data saved to: {output_file}')
    return df


def batch_tokenize(sentences, tokenizer, device):
    """Tokenize a batch of sentences."""
    return tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)


def generate_labels(output_filename = 'train_df_labels.csv'):
    """
    Generate GoEmotion labels for each sentence in each story.
    Count how many sentences of each of the 28 labels occurred for each story.
    Write results to a csv file.
    """
    train_df = get_dataframe('train', 'target')

    labels = ['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire','disappointment',\
     'disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy','love','nervousness',\
     'optimism','pride','realization','relief','remorse','sadness','surprise','neutral']


    model = AutoModelForSequenceClassification.from_pretrained('monologg/bert-base-cased-goemotions-original')
    tokenizer = AutoTokenizer.from_pretrained('monologg/bert-base-cased-goemotions-original')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with open(output_filename, 'w', newline='') as file:
        writer = csv.writer(file)

        story_labels = []
        for story in tqdm(train_df['Prompt']): # to test on small dataset, change to train_df['Prompt'][:100]
            story = story.lower()
            sentences = re.split(r'[\!\?\.] |[\!\?\.]$', story)

            counts_labels = {item: 0 for item in labels}
            batch_size = 8
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                inputs = batch_tokenize(batch, tokenizer, device)
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)

                top_idxs = torch.argmax(probs, dim=1)

                for top_idx in top_idxs:
                    counts_labels[labels[top_idx]] += 1
            story_labels.append(counts_labels)
            writer.writerow([story, counts_labels])
    print(f'Data with 28 labels: {output_filename}')

def map_labels_to_quadrants(quadrants_filename = 'data/emotion_quadrants.txt', labeled_data_filename = 'train_df_labels.csv', output_filename = 'train_df_labels_recoded.csv'):
    """
    Map the GoEmotions labels to quadrant numbers (1, 2, 3, 4, or neutral).
    Calculate counts and percentages of each quadrant for each story.
    Write results to a csv file.
    """
    quadrants = pd.read_table(quadrants_filename, names=['label', 'arousal', 'valence', 'quadrant'])
    train_df = pd.read_csv(labeled_data_filename, names=['Prompt', 'label_counts'],
                           converters={'label_counts': ast.literal_eval})

    quadrant_dict = dict(zip(quadrants['label'].str.lower(), quadrants['quadrant']))
    quadrant_dict['neutral'] = 'neutral'

    q1_count = []
    q2_count = []
    q3_count = []
    q4_count = []
    neutral_count = []
    for idx, row in train_df.iterrows():
        label_counts = row['label_counts']
        new_label_counts = {key: 0 for key in quadrant_dict.values()}
        for label, count in label_counts.items():
            new_label = quadrant_dict[label]
            new_label_counts[new_label] += count
        q1_count.append(new_label_counts['Q1'])
        q2_count.append(new_label_counts['Q2'])
        q3_count.append(new_label_counts['Q3'])
        q4_count.append(new_label_counts['Q4'])
        neutral_count.append(new_label_counts['neutral'])

    train_df['q1_count'] = q1_count
    train_df['q2_count'] = q2_count
    train_df['q3_count'] = q3_count
    train_df['q4_count'] = q4_count
    train_df['neutral_count'] = neutral_count

    train_df['num_sentences'] = train_df['q1_count'] + \
                                train_df['q2_count'] + \
                                train_df['q3_count'] + \
                                train_df['q4_count'] + \
                                train_df['neutral_count']

    train_df['q1_percentage'] = train_df['q1_count'] / train_df['num_sentences']
    train_df['q2_percentage'] = train_df['q2_count'] / train_df['num_sentences']
    train_df['q3_percentage'] = train_df['q3_count'] / train_df['num_sentences']
    train_df['q4_percentage'] = train_df['q4_count'] / train_df['num_sentences']
    train_df['neutral_percentage'] = train_df['neutral_count'] / train_df['num_sentences']
    train_df.to_csv(output_filename)
    print(f'Data with 4 quadrant labels: {output_filename}')

def main():
    download_stories()
    generate_labels()
    map_labels_to_quadrants()

if __name__ == '__main__':
    main()
