import nltk
from nltk.corpus import wordnet as wn
nltk.download("omw-1.4")
nltk.download("wordnet")
from transformers import CLIPProcessor, CLIPModel
import torch
from check_gpu import check

def get_adjectives(limit=None):
    adjectives = set()
    for synset in wn.all_synsets("a"):
        for lemma in synset.lemmas():
            adjectives.add(lemma.name())
            if limit and len(adjectives) >= limit:
                return adjectives
    return adjectives

def get_nouns(limit=None):
    nouns = set()
    for synset in wn.all_synsets("n"):
        for lemma in synset.lemmas():
            nouns.add(lemma.name())
            if limit and len(nouns) >= limit:
                return nouns
    return nouns

def compute_embedding(num_words, is_adjective):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
  processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
  if is_adjective:
    words = list(get_adjectives(limit=num_words))
  else:
    words = list(get_nouns(limit=num_words)) 
  tokenized_words = processor(words, padding=True, return_tensors="pt").to(device)

  with torch.no_grad():
    clip_features = model.get_text_features(**tokenized_words)

  if is_adjective:
    torch.save(clip_features, "adjective_features.pt")
    with open("adjectives.txt", "w") as f:
      f.write(", ".join(words))
  else:
    torch.save(clip_features, "noun_features.pt")
    with open("nouns.txt", "w") as f:
      f.write(", ".join(words))

if __name__ == "__main__":
  num_adj = 100000
  num_nouns = 50000
  # compute_embedding(num_adj, True) # comment out one due to memory constraint
  compute_embedding(num_nouns, False)




