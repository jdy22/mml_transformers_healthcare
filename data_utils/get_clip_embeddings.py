import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

prompts = [
    "a computerized tomography",
    "a magnetic resonance image",
    "a computerized tomography of a spleen",
    "a magnetic resonance image of a spleen",
           ]

text_tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_embeddings = model.encode_text(text_tokens)

print(text_embeddings.shape)