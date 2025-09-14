📰 AI-Powered Headline Generator

Generate short, meaningful headlines from long news articles using state-of-the-art transformer models.

🔹 About

This project leverages Hugging Face transformer models to automatically create concise Headlines from longer articles or text.

It uses DistilBART

These models are already trained on large summarization datasets, so you don’t need to manually provide a dataset unless you want to fine-tune.

🔹 Features

Generates short, accurate headlines from long text

Cleans up weak ending words from the headline

Adjustable minimum and maximum headline length

Easy to reuse as a Python module or API

🔹 How it Works

Uses a seq2seq transformer model (DistilBART or T5) for summarization

Generates candidate headlines using beam search (num_beams=4)

Takes only the first sentence

Removes weak ending words to ensure concise headlines

By - Sarang A Thakare
