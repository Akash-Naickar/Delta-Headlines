import torch 
import pprint
import evaluate
import numpy as np


from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

pp = pprint.PrettyPrinter()

dataset = load_dataset('gopalkalpande/bbc-news-summary', split='train')

full_dataset = dataset.train_test_split(test_size=0.2, shuffle=True)

dataset_train = full_dataset['train']
dataset_valid = full_dataset['test']

print(dataset_train)
print(dataset_valid)

def find_longest_length(dataset):
    """
    Find the longest article and summary in the entire training set.
    """
    max_length = 0
    counter_4k = 0
    counter_2k = 0
    counter_1k = 0
    counter_500 = 0
    for text in dataset:
        corpus = [
            word for word in text.split()
        ]
        if len(corpus) > 4000:
            counter_4k += 1
        if len(corpus) > 2000:
            counter_2k += 1
        if len(corpus) > 1000:
            counter_1k += 1
        if len(corpus) > 500:
            counter_500 += 1
        if len(corpus) > max_length:
            max_length = len(corpus)
    return max_length, counter_4k, counter_2k, counter_1k, counter_500

longest_article_length, counter_4k, counter_2k, counter_1k, counter_500 = find_longest_length(dataset_train['Articles'])
print(f"Longest article length: {longest_article_length} words")
print(f"Artciles larger than 4000 words: {counter_4k}")
print(f"Artciles larger than 2000 words: {counter_2k}")
print(f"Artciles larger than 1000 words: {counter_1k}")
print(f"Artciles larger than 500 words: {counter_500}")
longest_summary_length, counter_4k, counter_2k, counter_1k, counter_500 = find_longest_length(dataset_train['Summaries'])
print(f"Longest summary length: {longest_summary_length} words")
print(f"Summaries larger than 4000 words: {counter_4k}")
print(f"Summaries larger than 2000 words: {counter_2k}")
print(f"Summaries larger than 1000 words: {counter_1k}")
print(f"Summaries larger than 500 words: {counter_500}")

def find_avg_sentence_length(dataset):
    """
    Find the average sentence in the entire training set.
    """
    sentence_lengths = []
    for text in dataset:
        corpus = [
            word for word in text.split()
        ]
        sentence_lengths.append(len(corpus))
    return sum(sentence_lengths)/len(sentence_lengths)

avg_article_length = find_avg_sentence_length(dataset_train['Articles'])
print(f"Average article length: {avg_article_length} words")
avg_summary_length = find_avg_sentence_length(dataset_train['Summaries'])
print(f"Averrage summary length: {avg_summary_length} words")

MODEL = 't5-base'
BATCH_SIZE = 2
NUM_PROCS = 1
EPOCHS = 1
OUT_DIR = 'results_t5base'
MAX_LENGTH = 256 # Maximum context length to consider while preparing dataset.

tokenizer = T5Tokenizer.from_pretrained(MODEL)

# Function to convert text data into model inputs and targets
def preprocess_function(examples):
    inputs = [f"summarize: {article}" for article in examples['Articles']]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length'
    )

    # Set up the tokenizer for targets
    targets = [summary for summary in examples['Summaries']]
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length'
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the function to the whole dataset
tokenized_train = dataset_train.map(
    preprocess_function,
    batched=True,
    num_proc=NUM_PROCS
)
tokenized_valid = dataset_valid.map(
    preprocess_function,
    batched=True,
    num_proc=NUM_PROCS
)

model = T5ForConditionalGeneration.from_pretrained(MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions[0], eval_pred.label_ids

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        use_stemmer=True, 
        rouge_types=[
            'rouge1', 
            'rouge2', 
            'rougeL'
        ]
    )

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=OUT_DIR,
    logging_steps=10,
    evaluation_strategy='steps',
    eval_steps=200,
    save_strategy='epoch',
    save_total_limit=2,
    report_to='tensorboard',
    learning_rate=0.0001,
    dataloader_num_workers=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics
)

history = trainer.train()

tokenizer.save_pretrained(OUT_DIR)

