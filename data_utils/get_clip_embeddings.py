import torch
import clip
import pickle


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    ct_pos_prompts = [
        "a computerized tomography of a spleen",
        "a computerized tomography of a right kidney",
        "a computerized tomography of a left kidney",
        "a computerized tomography of a gall bladder",
        "a computerized tomography of an esophagus",
        "a computerized tomography of a liver",
        "a computerized tomography of a stomach",
        "a computerized tomography of an aorta",
        "a computerized tomography of an inferior vena cava",
        "a computerized tomography of a pancreas",
        "a computerized tomography of a right adrenal gland",
        "a computerized tomography of a left adrenal gland",
        "a computerized tomography of a duodenum",
        "a computerized tomography of a bladder",
        "a computerized tomography of a prostrate or uterus",
            ]

    ct_neg_prompts = [
        "a computerized tomography without a spleen",
        "a computerized tomography without a right kidney",
        "a computerized tomography without a left kidney",
        "a computerized tomography without a gall bladder",
        "a computerized tomography without an esophagus",
        "a computerized tomography without a liver",
        "a computerized tomography without a stomach",
        "a computerized tomography without an aorta",
        "a computerized tomography without an inferior vena cava",
        "a computerized tomography without a pancreas",
        "a computerized tomography without a right adrenal gland",
        "a computerized tomography without a left adrenal gland",
        "a computerized tomography without a duodenum",
        "a computerized tomography without a bladder",
        "a computerized tomography without a prostrate or uterus",
            ]

    mri_pos_prompts = [
        "a magnetic resonance image of a spleen",
        "a magnetic resonance image of a right kidney",
        "a magnetic resonance image of a left kidney",
        "a magnetic resonance image of a gall bladder",
        "a magnetic resonance image of an esophagus",
        "a magnetic resonance image of a liver",
        "a magnetic resonance image of a stomach",
        "a magnetic resonance image of an aorta",
        "a magnetic resonance image of an inferior vena cava",
        "a magnetic resonance image of a pancreas",
        "a magnetic resonance image of a right adrenal gland",
        "a magnetic resonance image of a left adrenal gland",
        "a magnetic resonance image of a duodenum",
        "a magnetic resonance image of a bladder",
        "a magnetic resonance image of a prostrate or uterus",
            ]

    mri_neg_prompts = [
        "a magnetic resonance image without a spleen",
        "a magnetic resonance image without a right kidney",
        "a magnetic resonance image without a left kidney",
        "a magnetic resonance image without a gall bladder",
        "a magnetic resonance image without an esophagus",
        "a magnetic resonance image without a liver",
        "a magnetic resonance image without a stomach",
        "a magnetic resonance image without an aorta",
        "a magnetic resonance image without an inferior vena cava",
        "a magnetic resonance image without a pancreas",
        "a magnetic resonance image without a right adrenal gland",
        "a magnetic resonance image without a left adrenal gland",
        "a magnetic resonance image without a duodenum",
        "a magnetic resonance image without a bladder",
        "a magnetic resonance image without a prostrate or uterus",
            ]

    ct_pos_tokens = clip.tokenize(ct_pos_prompts).to(device)
    ct_neg_tokens = clip.tokenize(ct_neg_prompts).to(device)
    mri_pos_tokens = clip.tokenize(mri_pos_prompts).to(device)
    mri_neg_tokens = clip.tokenize(mri_neg_prompts).to(device)
    with torch.no_grad():
        ct_pos_embeddings = model.encode_text(ct_pos_tokens)
        ct_neg_embeddings = model.encode_text(ct_neg_tokens)
        mri_pos_embeddings = model.encode_text(mri_pos_tokens)
        mri_neg_embeddings = model.encode_text(mri_neg_tokens)

    print(ct_pos_embeddings.shape)
    print(ct_neg_embeddings.shape)
    print(mri_pos_embeddings.shape)
    print(mri_neg_embeddings.shape)


if __name__ == "__main__":
    main()