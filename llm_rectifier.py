import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm

# =====================
# Config
# =====================
DATA_DIR = "output_files"
INPUT_FILE = os.path.join(DATA_DIR, "modified_files.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "llm_rectified_message.csv")
PLOT_FILE = os.path.join(DATA_DIR, "hit_rates.png")

# Models
GEN_MODEL_NAME = "mamiksik/CommitPredictorT5"      # generator
RECTIFIER_MODEL_NAME = "google/flan-t5-base"       # rectifier
EVAL_MODEL_NAME = "facebook/bart-large-mnli"       # evaluator (zero-shot classifier)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================
# Load Models
# =====================
print("Loading models...")
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(device)

rect_tokenizer = AutoTokenizer.from_pretrained(RECTIFIER_MODEL_NAME)
rect_model = AutoModelForSeq2SeqLM.from_pretrained(RECTIFIER_MODEL_NAME).to(device)

# Evaluator uses HuggingFace pipeline (zero-shot classification)
evaluator = pipeline("zero-shot-classification", model=EVAL_MODEL_NAME, device=0 if torch.cuda.is_available() else -1)

print("✅ Models loaded successfully")

# =====================
# Generator (first LLM)
# =====================
def generate_commit_message(diff_text, max_length=64):
    if not isinstance(diff_text, str) or len(diff_text.strip()) == 0:
        return ""
    inputs = gen_tokenizer(
        diff_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    ).to(device)
    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# =====================
# Rectifier (second LLM)
# =====================
def llm_rectifier(original_msg, llm_msg, filename, diff_text, max_length=64):
    if not llm_msg.strip():
        return original_msg.strip()

    diff_snippet = (diff_text or "")[:600]
    prompt = f"""
You are a commit-message rectifier for per-file changes.

Context:
- Developer message: "{original_msg}"
- Previous LLM suggestion: "{llm_msg}"
- Filename: "{filename}"
- File diff snippet:
\"\"\"{diff_snippet}\"\"\"

Task:
Write ONE clear, concise commit message (<15 words) in imperative style.
If bug fix, start with "Fix".
If tests, start with "Update tests".
If improvement, use "Improve" or "Refactor".
"""

    inputs = rect_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = rect_model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

    rectified = rect_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if not rectified:
        rectified = llm_msg.strip() or original_msg.strip()
    return rectified

# =====================
# Evaluator (third LLM)
# =====================
def is_bugfix_message(msg):
    """
    Uses zero-shot classification to judge if a commit message
    describes a bug fix.
    """
    if not isinstance(msg, str) or not msg.strip():
        return False

    result = evaluator(
        msg,
        candidate_labels=[
            "bug fix", "fix", "bug", "error correction",
            "patch", "defect correction", "handle issue",
            "feature", "refactor", "test update"
        ],
        hypothesis_template="This commit is a {}."
    )
    top_label = result["labels"][0].lower()
    if top_label in ["bug fix", "fix", "bug", "error correction", "patch", "defect correction", "handle issue"]:
        return True
    BUG_KEYWORDS = ["fix", "bug", "error", "issue", "defect", "patch", "handle"]
    if any(word in msg.lower() for word in BUG_KEYWORDS):
        return True
    return False


def evaluate(df):
    total = len(df)
    if total == 0:
        return {"RQ1": 0, "RQ2": 0, "RQ3": 0}

    rq1_hits = sum(is_bugfix_message(msg) for msg in tqdm(df["Message"], desc="Evaluating RQ1"))
    rq2_hits = sum(is_bugfix_message(msg) for msg in tqdm(df["LLM Inference (fix type)"], desc="Evaluating RQ2"))
    rq3_hits = sum(is_bugfix_message(msg) for msg in tqdm(df["Rectified Message"], desc="Evaluating RQ3"))

    return {
        "RQ1 (Developer msg hit rate)": rq1_hits / total,
        "RQ2 (LLM msg hit rate)": rq2_hits / total,
        "RQ3 (Rectifier msg hit rate)": rq3_hits / total,
    }

# =====================
# Plot Evaluation
# =====================
def plot_hit_rates(hit_rates):
    plt.figure(figsize=(7, 4.5))
    labels = list(hit_rates.keys())
    values = [v * 100 for v in hit_rates.values()]
    bars = plt.bar(labels, values, color=["skyblue", "salmon", "lightgreen"])
    plt.ylabel("Hit Rate (%)")
    plt.title("Commit Message Hit Rates (RQ1–RQ3)")
    plt.ylim(0, 100)
    for i, v in enumerate(values):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.show()
    print(f"📊 Plot saved to {PLOT_FILE}")

# =====================
# Main
# =====================
def main():
    print(f"Reading input file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    if "Diff" not in df.columns or "Message" not in df.columns:
        raise ValueError("Input CSV must contain 'Diff' and 'Message' columns.")

    llm_outputs, rectified_outputs = [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        diff_text = str(row.get("Diff", ""))
        original_msg = str(row.get("Message", ""))
        filename = str(row.get("Filename", "unknown"))

        try:
            llm_msg = generate_commit_message(diff_text)
        except Exception as e:
            print(f"[Generator error row {idx}] {e}")
            llm_msg = ""

        try:
            rectified_msg = llm_rectifier(original_msg, llm_msg, filename, diff_text)
        except Exception as e:
            print(f"[Rectifier error row {idx}] {e}")
            rectified_msg = llm_msg or original_msg or ""

        llm_outputs.append(llm_msg)
        rectified_outputs.append(rectified_msg)

    df["LLM Inference (fix type)"] = llm_outputs
    df["Rectified Message"] = rectified_outputs

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Results written to {OUTPUT_FILE}")

    results = evaluate(df)
    print("\n📊 Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.2%}")

    plot_hit_rates(results)

if __name__ == "__main__":
    main()
    

