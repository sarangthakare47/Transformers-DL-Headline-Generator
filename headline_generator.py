from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

datafor_model = "sshleifer/distilbart-xsum-12-6"
tokenizer = AutoTokenizer.from_pretrained(datafor_model)
model = AutoModelForSeq2SeqLM.from_pretrained(datafor_model)

headline_generator = pipeline("summarization", model=model, tokenizer=tokenizer)

ignore_end = {"as", "will", "to", "of", "in", "for", "on", "at", "with", "by", "and"}

def generate_headline(article: str, max_len=10, min_len=3):
    result = headline_generator(
        article,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        num_beams=4,
        early_stopping=True
    )
    
    headline_text = result[0]['summary_text'].split(".")[0].strip()

    words = headline_text.split()
    while words and words[-1].lower() in ignore_end:
        words = words[:-1]
    
    return " ".join(words)

if __name__ == "__main__":
    print("📰 Headline Generator (Short Headlines Mode)")
    print("Paste your article text (press Enter twice to finish):\n")

    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    article_text = " ".join(lines)

    if article_text.strip():
        headline = generate_headline(article_text)
        print("\n✅ Generated Headline:\n", headline)
    else:
        print("⚠️ No text entered.")
