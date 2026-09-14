# CLAUDE.md — AI in Engineering (MEEN41490) Course Website

This file holds project context for building the course webpage that will host
material for **AI in Engineering (MEEN41490)**. It does not contain writing-style
or text-generation rules — those are supplied separately, per activity, when
drafting actual content.

## 1. What this project is

Juan Pablo is building web-hosted material for a UCD module he supports. The
site's job is to present the practicals, exercises, and supporting content
developed for the module in a form students can use. Material is created
practical by practical, on request — this file exists so each new practical is
built with the right sense of what came before it and what comes after.

## 2. Module facts (from the UCD module page)

- **Code / Title:** MEEN41490 — AI in Engineering
- **Info page:** https://hub.ucd.ie/usis/!W_HU_MENU.P_PUBLISH?p_tag=MODULE&MODULE=MEEN41490
- **School:** Mechanical & Materials Engineering, College of Engineering & Architecture
- **Level:** 4 (Master's) · **Credits:** 5 · **Trimester:** Autumn · **Delivery:** On campus
- **Module Coordinator:** Dr David Garzón Ramos
- **Official description:** A project- and practice-based module giving students the
  toolkit to specify, commission, and evaluate AI systems in engineering and
  enterprise contexts — where AI is worth applying, what it takes to work, what it
  can realistically deliver. Teams develop a proposal for an AI-based solution to a
  societal problem, weighing sustainability and feasibility.
- **Learning outcomes:** evaluate whether AI is viable for an engineering problem;
  understand engineering data types and the models built from them; build
  evidence-based arguments about AI system claims; assess ethical/legal/professional
  obligations of applying AI; devise sustainable AI solutions aligned with the UN
  SDGs and communicate them to mixed-expertise audiences.
- **Indicative content:** approaches to AI (symbolic, discriminative, generative,
  physical); tools for building AI solutions (preparing data, running pretrained
  models); commercialisation potential; AI regulation and data protection;
  technical-feasibility assessment.
- **Learning recommendation on file:** "basic programming and statistics" — in
  practice this cohort has **no programming background**, so all practical
  material must be built for that starting point regardless of the catalogue note.
- **Effort hours:** 36 lectures, 40 specified learning activities, 50 autonomous
  learning, 126 total.
- **AI-use policy for students:** critical engagement with AI tools is encouraged
  (brainstorming, summarising, testing ideas); any use must be cited.

### Assessment breakdown

| Component | Timing | % of grade |
|---|---|---|
| In-class test | Week 10 | 25 |
| Guest-session exercises (Wk 3, 7, 12) | Weeks 3/7/12 | 15 |
| Pitch (3 min, slides) | Week 4 | 5 |
| Data collection plan | Week 6 | 5 |
| Poster (presented + defended) | Week 11 | 20 |
| Final team report | Week 15 | 30 |

All group work builds toward one running team project: a proposed AI-based
solution to a sustainability challenge (pitch → data plan → poster → report).

## 3. Audience and constraints

- Master's students, **zero prior programming experience** — this is the binding
  constraint on every practical.
- All coding activities must be simple, guided, and runnable without local setup.
- Primary tools: **Google Colab** (notebooks).
- Material should scaffold toward the team project deliverables above, not just
  teach isolated technique.

## 4. Module schedule (Autumn trimester)

| Wk | Mon lecture | Fri practical | Due this week |
|---|---|---|---|
| 1 | Module intro; what AI is and is not | Project intro, worked example, team formation | |
| 2 | Symbolic AI | Introduction to tools and working environment | |
| 3 | Data as raw material | Guest specialist session | Exercise, in class |
| 4 | How a model learns | Project session: idea checkpoint, pitch prep | Pitch video |
| 5 | Perception | Getting data into a notebook; demonstration | |
| 6 | Simulation and machine learning | Running a pretrained model; demonstration | Data collection plan |
| 7 | Generative models | Guest specialist session | Exercise, in class |
| 8 | No lecture (public holiday); video lecture on Brightspace | Training a small model; demonstration | |
| 9 | Generative AI at work | Making a model usable by others; demonstration | |
| 10 | Physical AI | In-class test; demonstration | In-class test |
| 11 | Adoption and governance | Poster session | Poster, submitted Wed 18 Nov |
| 12 | Collective intelligence | Guest specialist session | Exercise, in class |
| 13 | Revision week | | |
| 14 | Examinations | | |
| 15 | Examinations | | Report, Fri 18 Dec |

## 5. How to use this file when building a practical

When asked to develop material for a specific practical:

1. Locate it in the schedule above and read off the lecture it pairs with.
2. Check the 1–2 practicals **before** it — assume students already have that
   background (tools, vocabulary, prior notebook patterns) and don't re-teach it.
3. Check what's **coming up** — e.g., Wk5–6 (getting data in, running a pretrained
   model) sets up Wk8 (training a small model) and Wk9 (making a model usable by
   others); Wk2 (tools/environment) must leave students ready for Wk5 onward.
4. Note whether the practical feeds a graded deliverable (see assessment table)
   and, if so, build toward that output explicitly.
5. Keep every activity workable in Colab/Hugging Face by someone who has never
   written code before this module.

## 6. Prior related work (context, not necessarily reusable as-is)

- A previous, separate offering of an "AI in Engineering" course (undergraduate,
  not this master's module) used a Hugging Face org `aiengineering0926` and
  explored: a photogrammetry + LLM fabrication-analysis lab, interactive Gradio
  sandboxes, Gradio Lite/Static Spaces, KNIME, and Google Teachable Machine as
  no-code/beginner-friendly ML tools.
- Hugging Face removed free CPU Basic compute for Gradio/Docker Spaces in
  mid-2026; ZeroGPU was explored as an alternative. Worth re-checking current
  Spaces pricing/limits before committing any practical to a specific Spaces tier.
- Whether any of the above is reused for MEEN41490 is undecided — confirm before
  building on it, since that prior course had a different (undergraduate) audience.

## 7. Open items for the webpage itself

Not yet decided — resolve before/while building site structure:
- Site scope: every practical, or only the ones needing custom material?
- Structure: one page per week, per practical, or a continuous scroll?
- Hosting/build target (static site, GitHub Pages, etc.) — not yet specified.