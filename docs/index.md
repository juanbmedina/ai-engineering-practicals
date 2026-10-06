---
layout: default
title: "AI in Engineering — MEEN41490"
---

# AI in Engineering — MEEN41490

Practical sessions for MEEN41490, AI in Engineering. Each practical below pairs
a walkthrough page with a Google Colab notebook.

> **New to Google Colab?** Start with [Getting Started with Google Colab]({{ '/guides/getting-started-colab/' | relative_url }}) before your first practical.

## Practicals

<ul>
{% for p in site.data.practicals %}
  <li><a href="{{ p.path | relative_url }}">Practice {{ p.practice }}: {{ p.title }}</a></li>
{% endfor %}
</ul>

## Demonstrations

Demonstrations are ready-made apps that run in Google Colab. They let you explore an AI technique without writing any code.

> **Some demonstrations need an API key.** Follow [Getting a Free API Key for Google Gemini and Groq]({{ '/guides/api-keys-groq-gemini/' | relative_url }}) before you open them.

<ul>
{% for d in site.data.demos %}
  <li><a href="{{ d.path | relative_url }}">{{ d.title }}</a></li>
{% endfor %}
</ul>
