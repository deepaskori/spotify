import pandas as pd
from transformers import AutoTokenizer,DataCollatorForSeq2Seq
import psycopg2
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from evaluate import load
import numpy as np
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
nltk.download('punkt')


conn = psycopg2.connect(
    database="postgres",
    user='postgres',
    password='123Deepa',
    port='5432'
)



df = pd.read_sql_query('SELECT * FROM spotify',con=conn) 
df = df.iloc[:10000, :] 
spotify_dataset = Dataset.from_pandas(df) 

spotify_dataset = spotify_dataset.train_test_split(test_size=0.2)
# print("spotify_dataset", spotify_dataset)

print(spotify_dataset["train"][0])

metric = load("rouge")
# print(metric)

checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prefix = "summarize: "


max_input_length = 1024
max_target_length = 128

def spotify_preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["tracks"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    labels = tokenizer(text_target=examples["playlist"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# print("preprocessing example", spotify_preprocess_function(spotify_dataset['train'][:2]))
spotify_tokenized_datasets = spotify_dataset.map(spotify_preprocess_function, batched = True)


model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


batch_size = 16
model_name = checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    "finetuned-spotify-t5",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    result = {key: value * 100 for key, value in result.items()}
    
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=spotify_tokenized_datasets["train"],
    eval_dataset=spotify_tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.push_to_hub()








