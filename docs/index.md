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
  <li><a href="{{ p.path | relative_url }}">Week {{ p.week }}: {{ p.title }}</a></li>
{% endfor %}
</ul>
