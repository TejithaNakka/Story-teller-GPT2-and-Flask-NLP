import csv

def format_stories_for_gpt2(csv_file_path, output_file_path):
    """
    Formats story data from a CSV file into a text file suitable for GPT-2 fine-tuning.
    Each story is separated by the special token '<|endoftext|>'.

    Args:
        csv_file_path (str): Path to the CSV file containing the stories.
        output_file_path (str): Path to the output text file.
    """

    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:

        reader = csv.DictReader(csvfile)
        for row in reader:
            outfile.write(f"<|startoftext|>\n")  # Beginning of story
            outfile.write(f"[Genre: {row['Genre']}]\n")
            outfile.write(f"[Prompt: {row['Prompt']}]\n")
            outfile.write(row['Story'] + "\n")  # The actual story content
            outfile.write(f"<|endoftext|>\n")  # End of story marker

# Example usage:
#csv_file_path = r'C:\Users\tejitha.sai.s.nakka\Desktop\stories_dataset.csv' 
# The above path might be incorrect.
# Verify the file 'stories_dataset.csv' exists at the specified location.
# If not, update the path below accordingly.
# If the file is in the same directory as your script, you can use just the filename:
csv_file_path = 'stories_dataset.csv'  
output_file_path = 'stories.txt'  # Choose a name for your output file
format_stories_for_gpt2(csv_file_path, output_file_path)
#88d5d235c48a62f71cee8298428d22d96700ffc1  api key for wandb
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 1. Load the Pre-trained Model and Tokenizer
model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add the special tokens for start and end of text
special_tokens_dict = {'eos_token': '<|endoftext|>', 'bos_token': '<|startoftext|>'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer)) 

# 2. Prepare the Dataset
train_path = 'stories.txt'  # Path to the text file you created
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=128,  # Adjust this based on your story lengths
)

# 3. Create a Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# 4. Set Up Training Arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-story-generator",  # Choose your output directory
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust as needed
    per_device_train_batch_size=4,  # Adjust based on your GPU memory
    save_steps=10_000,
    save_total_limit=2,
)

# 5. Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# 6. Fine-tune!
trainer.train()

# 7. Save the Model
trainer.save_model("./gpt2-finetuned-story-generator")  # Save your fine-tuned model
tokenizer.save_pretrained("./gpt2-finetuned-story-generator")