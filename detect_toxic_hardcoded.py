from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = 'unitary/toxic-bert'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("i love you", return_tensors="pt")
outputs = model(**inputs)

# The output logits can be converted to probabilities using a softmax function
probs = outputs[0].softmax(1)

# If the model was trained to predict multiple classes (e.g., toxic, severe_toxic, obscene, threat, insult, identity_hate),
# then `probs` will be a tensor of probabilities for each of these classes.

print(probs)

classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Convert tensor to list for iteration
# The detach().numpy() function is used to convert the tensor into a numpy
# array, and then into a Python list for easy iteration.
probs_list = probs.detach().numpy()[0].tolist()

# Print each class and its corresponding probability
for i in range(len(classes)):
    # print(f"{classes[i]}: {probs_list[i]:.2f}%")
    print(f"{classes[i]}: {probs_list[i]*100:.2f}%")