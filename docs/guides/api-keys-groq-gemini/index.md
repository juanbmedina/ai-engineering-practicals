---
layout: default
title: "Getting a Free API Key for Google Gemini and Groq"
---

# Getting a Free API Key for Google Gemini and Groq

Some demonstrations in this module send text to a Large Language Model (LLM) that runs on the servers of an AI provider. The provider needs an Application Programming Interface (API) key to know who sends each request. An API key is a long personal password, and both providers below give one for free.

You need only **one** key. We recommend Gemini first, because it gives the most reliable results in the demonstrations of this module. Create a Groq key as well if you want to compare the two providers.

> **Keep your key private.** Anyone who has your key can spend your free quota. Paste the key only into the *Your API Key* box of the app, and never into a notebook cell, an email or a shared document.

## Option A — Google Gemini

1. Open **[aistudio.google.com/apikey](https://aistudio.google.com/apikey)** and sign in with a Google account.
2. Accept the terms of service the first time you open Google AI Studio.
3. Check whether the page already lists a key. For new users, Google AI Studio often creates a default project and a first key automatically.
4. If the list is empty, click **Create API key**. Then copy the key that appears.

## Option B — Groq

1. Open **[console.groq.com/keys](https://console.groq.com/keys)**.
2. Sign in with a Google account, a GitHub account or an email address. Groq does not ask for a credit card on its free plan.
3. Click **Create API Key** and give the key a name, e.g., `MEEN41490`.
4. Copy the key immediately and store it in a safe place, e.g., a password manager or a private note. Groq shows each key **only once**. After you close the window, you cannot read the key again.

A Groq key starts with `gsk_`.

## Use the key in a demonstration

1. Run the notebook of the demonstration until the app appears.
2. In **AI Provider**, choose the provider that issued your key. A Groq key does not work with Gemini, and a Gemini key does not work with Groq.
3. Paste the key into **Your API Key**. Then press **Analyze**.

The app does not store your key. Consequently, you paste the key again every time you restart the notebook.

## When something goes wrong

| Problem | Meaning | What to do |
|---|---|---|
| `401 Unauthorized` | The key is wrong, or it belongs to the other provider. | Copy the key again, and check the provider selected in the app. |
| `429 Too Many Requests` | You reached the free limit for this minute or for this day. | Wait one minute, or switch to the other provider. |
| You lost the Groq key | Groq shows each key only once. | Delete the old key in the console, and create a new one. |
| Your key became public | Somebody else can spend your quota. | Delete the key in the console straight away, and create a new one. |

Each provider publishes its current free limits on its own website, and those limits change over time. For the demonstrations in this module, the free limits are normally enough.
