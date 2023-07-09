from transformers import DistilBertTokenizer, DistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

# The output embeddings is a tensor of shape (batch_size, sequence_length, hidden_size) which contains the last
# hidden states of the model. batch_size is the number of sentences, sequence_length is the number of tokens in each
# sentence, and hidden_size is the size of the hidden layers in the model. In this case, the batch_size is 1 because
# you've only processed one sentence.

# Outputs will be a tuple where the first element is the embeddings.
embeddings = outputs[0]

print(embeddings)
