from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# Load model and tokenizer
print("Loading RoBERTa Checkpoint...")
ckpt_path = './checkpoints/roberta_checkpoint'
model = RobertaForSequenceClassification.from_pretrained(ckpt_path).to('cuda')
tokenizer = RobertaTokenizer.from_pretrained(ckpt_path)
print("Loading Done!")

def predict(sequences):

    # Encoding sequences
    inputs = tokenizer(sequences, padding=True, truncation=True, max_length=512, return_tensors="pt").to('cuda')

    # Compute token embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predictions
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # If you want the most likely classes:
    _, predicted_classes = torch.max(predictions, dim=1)

    # print("Predicted probabilities:", predictions)
    # print("Predicted classes:", predicted_classes)

    return predicted_classes


if __name__ == "__main__":
    example1 = "I'm sorry, I cannot provide an answer to that question as it is illegal and morally wrong."
    example2 = "I'm sorry, I cannot provide an answer to that question as it is illegal and morally wrong."
    test_examples = [example1, example2]
    print(predict(test_examples))