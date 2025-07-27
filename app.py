import os
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/transformers"
os.environ["XDG_CACHE_HOME"] = "/tmp/huggingface/.cache"  # <- critical
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface/.cache/huggingface/hub"  # <- important

# Optional safety for datasets
os.environ["HF_DATASETS_OFFLINE"] = "0"
import io
import base64
import torch
import warnings
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify
from datasets import load_dataset
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, AutoencoderKL
from sentence_transformers import SentenceTransformer, util



# Initialize Flask app
app = Flask(__name__)

# ========== Load resources ==========

print("Loading fashion image dataset...")
dataset = load_dataset("duyngtr16061999/fashion_text_to_image", split="train")
df = dataset.to_pandas()

print("Loading diffusion model...")
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float32
).to("cpu")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cpu")
pipe.vae = vae

print("Loading sentence transformer (CLIP)...")
clip_model = SentenceTransformer("clip-ViT-B-32")

print("Loading fashion Q&A dataset...")
qa_dataset = load_dataset("lusabo/fashion_questions_answers", split="train")
qa_questions = [item["question"] for item in qa_dataset]
qa_embeddings = clip_model.encode(qa_questions, convert_to_tensor=True)

# ========== Helper Functions ==========

def find_best_match(prompt, df):
    prompt_emb = clip_model.encode(prompt, convert_to_tensor=True)
    for col in ["TEXT", "text", "ENG_TEXT", "blip2_caption1"]:
        if col in df.columns:
            df_embs = clip_model.encode(df[col].tolist(), convert_to_tensor=True)
            scores = util.cos_sim(prompt_emb, df_embs)[0]
            best_idx = torch.argmax(scores).item()
            return df.iloc[best_idx]
    return None

# ========== Flask Routes ==========

@app.route("/")
def home():
    return "✅ Fashion chatbot backend is running."

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    user_prompt = data.get("prompt", "").strip()

    if not user_prompt:
        return jsonify({"error": "Empty prompt"}), 400

    # Default values
    qa_answer = "Sorry, I couldn't find a related fashion tip."
    image_data = None
    image_url = None

    try:
        # === Find text answer ===
        prompt_emb = clip_model.encode(user_prompt, convert_to_tensor=True)
        qa_scores = util.cos_sim(prompt_emb, qa_embeddings)[0]
        qa_best_idx = torch.argmax(qa_scores).item()
        qa_answer = qa_dataset[qa_best_idx]["answer"]
    except Exception as e:
        print("Text match error:", e)

    try:
        # === Generate image ===
        generator = torch.Generator("cpu").manual_seed(42)
        with torch.inference_mode():
            result = pipe(
                prompt=user_prompt,
                negative_prompt="blurry, bad anatomy, watermark, extra hands",
                height=512,
                width=512,
                num_inference_steps=25,
                guidance_scale=7.5,
                generator=generator,
            )
        img = result.images[0]
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        image_data = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    except Exception as e:
        print("Image generation error:", e)

    try:
        # === Find matched image ===
        sample = find_best_match(user_prompt, df)
        image_url = sample.get("URL") if sample is not None else None
    except Exception as e:
        print("Image match error:", e)

    return jsonify({
        "answer": qa_answer,
        "image_url": image_url,
        "generated_image": image_data  # Base64-encoded PNG
    })

# ========== Run Server (required for HF Spaces) ==========

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)