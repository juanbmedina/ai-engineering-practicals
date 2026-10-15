---
layout: default
title: "Demonstration — Sentiment Analysis and Word Clouds with a Large Language Model"
---

# Demonstration — Sentiment Analysis and Word Clouds with a Large Language Model

## What the demonstration does

This demonstration reads one or more web pages and sends their text to a Large Language Model (LLM). The LLM classifies the overall tone of the pages as positive, negative or neutral. Then, the app draws a word cloud of the most frequent words, coloured green, red or grey according to that tone.

The analysis is cumulative, i.e., each new result includes the text of every page read so far. Consequently, you can watch how each new page shifts the overall tone and the most frequent words.

You do not write any code in this demonstration. The notebook holds the whole app behind three cells, and the app opens inside the notebook.

## Before you begin

1. Get a free API key from Google Gemini or Groq. The guide [Getting a Free API Key for Google Gemini and Groq]({{ '/guides/api-keys-groq-gemini/' | relative_url }}) takes about five minutes.
2. If Colab is new to you, read [Getting Started with Google Colab]({{ '/guides/getting-started-colab/' | relative_url }}) first.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Wig9Fc5PTo0SGmfdCb0v8XgEjWeRKhUk)

## Run the app

1. **Install the libraries.** Run cell 1 and wait about 30 seconds.
2. **Load the app.** Run cell 2. This cell holds the code of the whole app, and you can double-click its title to read it.
3. **Start the app.** Run cell 3. The app appears below the cell, and a public link that ends in `gradio.live` appears above it. Click that link to open the app in a new browser tab, where it has more space.
4. In the app, choose a provider, paste your key and paste one or more URLs, one per line. Then press **Analyze**.

The app has three sections. **1. Inputs** holds the provider, the model, your key and the URLs. **2. Processing** reports every step of the analysis, including every error. **3. Outputs** shows one word cloud for each page that the app could read.

## When something goes wrong

- For `401 Unauthorized` and `429 Too Many Requests`, see the table at the end of the [API key guide]({{ '/guides/api-keys-groq-gemini/' | relative_url }}).
- A `403` error on a URL means that the website blocks automated downloads. Try another page.
- A white box instead of the app means that your browser blocks the preview of Colab. Open the `gradio.live` link in a new tab instead.
- If the app stops responding, run cell 3 again.
- The public link works only while your notebook runs, and it expires after one week at most. Anyone with the link can open your app. However, each visitor must paste their own key.
