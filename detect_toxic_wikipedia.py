from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from colorama import Fore, Style, init

# Initialize colorama
init()

# Load the model and tokenizer
model_name = 'unitary/toxic-bert'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the dataset
# df = pd.read_csv('Training-Data/train-100.csv')
# skip over problematic rows instead of failing
df = pd.read_csv('Training-Data/train-1000.csv', on_bad_lines='skip')



# CSV columns:
# "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"

classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Create a dataframe to store probabilities
probs_df = pd.DataFrame(columns=classes)

# Create a dataframe to store binary ratings
ratings_df = df[classes]

# Iterate over comments in the dataset Wrap your DataFrame iterator (df.iterrows()) with tqdm to create a progress
# bar. The total=df.shape[0] argument is used to inform tqdm about the total number of iterations, which is equal to
# the number of rows in your DataFrame.
# Run your script in a terminal to see the tqdm progress bar, as it might not display correctly in some IDEs.
for index, row in tqdm(df.iterrows(), total=df.shape[0]):

    comment_text = row['comment_text']  # Replace 'comment_text' with the actual column name
    # inputs = tokenizer(comment_text, return_tensors="pt")
    # Truncate long comments, so we don't exceed the max tokens
    inputs = tokenizer(comment_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    probs_list = probs.detach().numpy()[0].tolist()

    # Append the probabilities to the dataframe
    probs_df.loc[index] = probs_list

# Now, 'probs_df' contains the probabilities, and 'ratings_df' contains the actual binary ratings.
# You can compare them as needed.

# Iterate over both dataframes
for index in range(len(probs_df)):
    # Check if any of the probabilities exceed 0.80
    if any(probs_df.loc[index] > 0.80):
        print(f"\nComment ID: {index + 1}")

        # Print the comment text in red
        print(Fore.RED + f"Comment: {df.loc[index, 'comment_text']}" + Style.RESET_ALL)

        print("Binary Ratings:")

        # Print the binary ratings
        for class_name in classes:
            print(f"{class_name}: {ratings_df.loc[index, class_name]}")

        print("\nModel Probabilities:")

        # Print the model's predicted probabilities
        for class_name in classes:
            print(f"{class_name}: {probs_df.loc[index, class_name] * 100:.2f}%")



