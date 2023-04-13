import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# load the adjectives and nouns features
adjective_features = torch.load("adjective_features.pt")
noun_features = torch.load("noun_features.pt")
# load the adjectives and nouns
with open("adjectives.txt", "r") as f:
    adjectives = f.read().split(", ")
with open("nouns.txt", "r") as f:
    nouns = f.read().split(", ")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_path = "/content/gdrive/MyDrive/CSC413/images/img.jpeg"
image = Image.open(image_path)
inputs = processor(images=[image], return_tensors="pt", padding=True)
preprocessed_image = inputs["pixel_values"].to(device)

# calculate the image and text features
with torch.no_grad():
    image_features = model.get_image_features(preprocessed_image)

def extract_top_k(image_features, text_features, words, k):
    similarities = (image_features @ text_features.T).squeeze(0)
    top_k_indices = similarities.topk(k).indices
    top_k_words = [words[idx] for idx in top_k_indices]
    return top_k_words

top_k_adjectives = 10
top_k_nouns = 5

top_adjectives = extract_top_k(image_features, adjective_features, adjectives, top_k_adjectives)
top_nouns = extract_top_k(image_features, noun_features, nouns, top_k_nouns)

print(top_adjectives)
print(top_nouns)

