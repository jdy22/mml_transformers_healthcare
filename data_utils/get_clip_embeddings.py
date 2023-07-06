import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

prompts = [
    "a computerized tomography",
    "a magnetic resonance image"
           ]

text_tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_embeddings = model.encode_text(text_tokens)

print(type(text_embeddings))
print(text_embeddings.shape)