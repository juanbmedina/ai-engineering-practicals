---
layout: default
title: "Practical 4 — Report generation using LLMs"
---

# Practical 4 — Report generation using LLMs

## How this session works

**This practical session has two sections, and both work with fictional data about an energy community.** The data describe the electricity use of five buildings of a neighbourhood that plans to produce and share solar energy. In Section 1, you run a demonstration that sends a summary of these data to a Large Language Model (LLM) on the servers of Groq or Google. That LLM writes the findings of an audit report. In Section 2, you download a small pretrained LLM, run it inside your own notebook, and send it exactly the same prompt. In two advanced tasks, you can then compare the two reports fact by fact.

Both sections use the same four data files, so the only difference between the two reports is the model that wrote them.

| Section | What you do | Notebook |
|---|---|---|
| 1 — Demonstration | Generate an audit report with an LLM that runs on the servers of a provider | Multimodal Audit Report Generator |
| 2 — Practice | Run a small LLM inside Colab on the same prompt, and compare the two reports | `MEEN41490_P4_Local_LLM_STUDENT` |

![Section 1 sends the prompt to the servers of a provider, whereas Section 2 runs a small model inside the Colab session]({{ '/practicals/audit_report_local_model/assets/where_the_model_runs.png' | relative_url }})

## The story behind the data

Ashgrove is a fictional neighbourhood with five buildings: two apartment blocks, a primary school, a corner shop and a community centre. The five buildings want to become an **energy community**, a group of neighbours who produce, share and sell energy together. The members plan to install solar panels on their roofs and to share the electricity among themselves. Then, they would sell the surplus to the grid and return the profit to the members.

Before anyone buys a solar panel, the community needs to know how much electricity each building uses, and in which months. Therefore, the first step of the transition was a metering campaign. Smart meters measured the monthly electricity use of each building from January to December 2025.

The coordinator of the future community now asks for an audit of these data, because they will size the shared solar panels and set the rules for sharing energy and profit. In this session, you play the role of the energy engineer who writes that audit with the help of an LLM. The coordinator also handed over three documents: a maintenance report, an alert from the electricity supplier and a note from the coordinator.

![The Ashgrove neighbourhood with five buildings, smart meters and planned solar panels, and the timeline of the metering campaign of 2025 with the four files of the session]({{ '/practicals/audit_report_local_model/assets/energy_community_story.png' | relative_url }})

The figure tells the story in two parts. Part 1 shows the five buildings, their smart meters and the solar panels that the community plans to share. Part 2 places the events of 2025 on a timeline and links each file of the session to the moment it describes.

## The data for the session

Download the four files of the session and save them in the same folder on your computer. You upload them in both sections. Every name, value, date and document in these files is invented for teaching.

- [`neighbourhood_electricity_2025.csv`]({{ '/practicals/audit_report_local_model/assets/data/neighbourhood_electricity_2025.csv' | relative_url }}){:download=""} — the electricity data
- [`maintenance_report_primary_school.pdf`]({{ '/practicals/audit_report_local_model/assets/data/maintenance_report_primary_school.pdf' | relative_url }}){:download=""} — the maintenance report
- [`supplier_high_consumption_notice.html`]({{ '/practicals/audit_report_local_model/assets/data/supplier_high_consumption_notice.html' | relative_url }}){:download=""} — the supplier alert
- [`community_coordinator_note.txt`]({{ '/practicals/audit_report_local_model/assets/data/community_coordinator_note.txt' | relative_url }}){:download=""} — the coordinator's note

### The electricity data — `neighbourhood_electricity_2025.csv`

This table holds the 60 monthly readings of the metering campaign, with one row for each building and each month of 2025. The readings use the kilowatt-hour (kWh) as their unit. One kWh is the energy that a 1 kW appliance uses in one hour.

The table below shows the first ten rows of the file, which cover January and February 2025.

| `month` | `building` | `building_type` | `floor_area_m2` | `electricity_consumption_kwh` | `reading_type` |
|---|---|---|---|---|---|
| 2025-01 | Apartment Block A | residential | 1900 | 9445 | measured |
| 2025-01 | Apartment Block B | residential | 1300 | 5929 | measured |
| 2025-01 | Primary School | education | 2100 | 13883 | measured |
| 2025-01 | Corner Shop | retail | 450 | 8890 | measured |
| 2025-01 | Community Centre | community | 800 | 3771 | measured |
| 2025-02 | Apartment Block A | residential | 1900 | 8260 | measured |
| 2025-02 | Apartment Block B | residential | 1300 | 5645 | measured |
| 2025-02 | Primary School | education | 2100 | 11429 | measured |
| 2025-02 | Corner Shop | retail | 450 | 8751 | measured |
| 2025-02 | Community Centre | community | 800 | 3376 | measured |

Each row has six columns.

| Column | Meaning | Example from the first row |
|---|---|---|
| `month` | The month of the reading, written as year and month | `2025-01` |
| `building` | The name of the building | `Apartment Block A` |
| `building_type` | The use of the building: residential, education, retail or community | `residential` |
| `floor_area_m2` | The floor area of the building, in square metres | `1900` |
| `electricity_consumption_kwh` | The electricity that the building used in that month, in kWh | `9445` |
| `reading_type` | `measured` for a reading of the smart meter, `estimated` for a value that the supplier estimated | `measured` |

In both sections, the column `electricity_consumption_kwh` is the column to analyse, and the column `building` groups the rows.

### The maintenance report — `maintenance_report_primary_school.pdf`

This one-page PDF report comes from the building services of Ashgrove, and it is dated 24 February 2025. It describes a fault in the heating system of the Primary School. The report explains what failed, when the technician repaired it and how much extra electricity the fault used.

### The supplier alert — `supplier_high_consumption_notice.html`

This web page comes from Northlight Electricity, a fictional electricity supplier, and it is dated 5 February 2025. It warns the Primary School about an unusually high meter reading in January 2025. Then, it compares that reading with the reading of January 2024.

### The coordinator's note — `community_coordinator_note.txt`

This plain-text note comes from the coordinator of the future energy community, and it is dated 15 January 2026. It summarises the plan of the community and the events of the metering campaign in a short list. It also states the **baseline** of the feasibility study. The baseline is the average monthly electricity use per building that the plan of the community assumed.

---

## Section 1 — Demonstration: a language model through an API

### What the demonstration does

The Multimodal Audit Report Generator turns a data table into a PDF audit report. First, the app computes statistics and charts for the column that you choose. Then, it sends the statistics and the text of your documents to an LLM through an Application Programming Interface (API)—that is, a service that answers requests over the internet. Finally, the app joins the statistics, the charts and the findings of the LLM into a PDF report that you can download.

You do not write any code in this section. The notebook holds the whole app behind three cells, and the app opens inside the notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/150dBGVyIlCl6EyyjyhIz5MxTlP-AzOl4)

### Running the app

1. **Install the libraries.** Run cell 1 and wait about 30 seconds.
2. **Load the app.** Run cell 2. This cell holds the code of the whole app, and you can double-click its title to read it.
3. **Start the app.** Run cell 3. The app appears below the cell, and a public link that ends in `gradio.live` appears above it. Click that link to open the app in a new browser tab, where it has more space.

### Generating your audit report

Fill the boxes of the app with the four files and the settings of the table below. The supporting documents sit in a section of the app that opens with a click.

| Box in the app | What to enter |
|---|---|
| Excel/CSV file | `neighbourhood_electricity_2025.csv` |
| Target column | `electricity_consumption_kwh` |
| Comparison column | `building` |
| Supporting documents, PDF files | `maintenance_report_primary_school.pdf` |
| Supporting documents, HTML files | `supplier_high_consumption_notice.html` |
| Supporting documents, TXT files | `community_coordinator_note.txt` |
| AI Provider and Your API Key | Groq or Gemini, and the key of that provider |
| AI role | `Energy Engineer` |
| Alert threshold for the mean | `5500`, the baseline of the feasibility study in the coordinator's note |

Then, click **Generate Report**. The charts and the findings of the LLM appear in the app, and the PDF report is ready to download. If an upload fails, click the sample row under the inputs, which loads the same four files and settings.

> **Keep your PDF report.** The first advanced task of Section 2 compares section 5 of this report, "Audit Findings (AI Inference)", with the reply of a small model. Download the report and keep it open.

---

## Section 2 — Practice: a pretrained language model inside the notebook

### What you will do in this section

In Section 1, the LLM ran on the servers of Groq or Google, and it returned the audit findings over the internet. You never saw the model itself, only the answer that it returned.

In this section, you download a small pretrained LLM and run it inside your own Colab session. A pretrained model is a model that somebody else has already trained. You only use it. That is, you give it an input and you read its output. The model of today is Qwen3-0.6B, which the Qwen team of Alibaba published openly on the Hugging Face Hub.

You will build the prompt of the demonstration from the same four files, and you will send it to this small model. Then, in two advanced tasks, you will compare its findings with your PDF report from Section 1 and test whether the model reads the documents.

| | Section 1 — Demonstration | Section 2 — Practice |
|---|---|---|
| Where the model runs | Servers of Groq or Google | Your Colab session |
| Model | A large model, e.g., `openai/gpt-oss-20b` | `Qwen/Qwen3-0.6B` |
| Size | About 21 billion parameters for `gpt-oss-20b` | About 0.6 billion parameters |
| API key | Required | Not required |
| Where your data goes | To the provider, over the internet | It stays in your Colab session |

> **How to use this section.** Work in the notebook, where every cell marked `# YOUR CODE HERE` is yours to fill. The comments of each cell tell you what to write and what you should see. **Try it yourself first.** If you are stuck, come back to the matching part of this page and click **Show the implementation**.

> **Before you start.** Open the notebook, save your own copy to Drive, and rename it to `MEEN41490_P4_YourName`. Keep the four downloaded files at hand, because Part 1 asks you to upload them.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vVRaj7OE7BOBh7VM_4Pu8dSpPon-np9X)

### Part 0 — The Basics

Read this part before you open the notebook, because every cell of the notebook relies on these four ideas.

**A pretrained model.** Training an LLM means adjusting its parameters on a very large collection of text, and it needs weeks of computing time. A pretrained model has already gone through that step, so you can download it and use it within a minute. In Week 8, you will train a small model yourself and see that step from the inside.

**The model size.** A parameter is one number that training adjusted. Qwen3-0.6B has about 0.6 billion parameters, which fill a file of more than one gigabyte. In contrast, `gpt-oss-20b`, the default model of the demonstration, has about 21 billion parameters. Larger models usually write better answers, but they need more memory and more time.

**The model and the tokens.** A token is a short word or a piece of a longer word. The model first reads the whole prompt as a list of tokens. Then, it writes its reply one token at a time, and each new token depends on the prompt and on every token before it.

**The app workflow.** The app does not send the 60 rows of the electricity table. Instead, it sends four statistics, the text of three documents and the task. Consequently, no model can see which building raises the mean, because the prompt holds no value per building.

### Part 1 — Setting up the notebook and loading the data

#### 1.1 Running the setup cell

The notebook asks Colab for a T4 GPU, which writes each answer in a few seconds. If Colab connects you to a CPU instead, everything still works, but each answer of the model takes one to two minutes.

Run the setup cell, which installs one small library and prepares the functions of this practical. The cell prints `Ready.` when it finishes.

#### 1.2 Uploading the four files of the session

A Colab notebook runs on a virtual machine in the cloud, not on your computer. Therefore, the notebook cannot read the files in your Downloads folder. Instead, you upload the files to the storage of the virtual machine, which Colab shows in the **Files** panel.

1. Click the **folder icon** in the left sidebar of Colab. The **Files** panel opens, as in the picture below.
2. Drag the four downloaded files from your computer onto the Files panel. Alternatively, click the **upload icon**, the first icon at the top of the panel, with an arrow that points up. Then, select the four files in the window that opens.
3. Colab warns that uploaded files disappear when the session ends. Click **OK**.
4. Wait until the four file names appear in the panel. The folder `sample_data` belongs to Colab, and it has nothing to do with this practical.

![The Files panel of Colab, opened from the folder icon in the left sidebar. The upload icon is the first icon at the top of the panel]({{ '/guides/getting-started-colab/assets/p24.png' | relative_url }})

The notebook then runs `check_uploaded_files()`, which prints `found` or `MISSING` for each of the four files. If a file is missing, upload it again with its original name and run the check once more.

> **Uploaded files are temporary.** Colab deletes them when the session ends or restarts. If you come back later, upload the four files again before you run Part 2.

### Part 2 — Model inputs

In this part, you rebuild the prompt of the demonstration step by step, with the same functions as the app.

#### 2.1 The data table

pandas is a Python library for working with tables of data, and the notebook loads it under the short name `pd`. Today, you use pandas only to read the CSV file of the electricity readings and to look at its first rows.

The command `pd.read_csv(...)` reads a CSV file of the Files panel into a table, which Python calls a DataFrame.

<details markdown="1">
<summary>Show the implementation</summary>

```python
data = pd.read_csv("neighbourhood_electricity_2025.csv")
data.head(10)
```

- `pd.read_csv("...")` opens the uploaded CSV file and turns it into a table. The name between the quotation marks must match the name in the Files panel exactly.
- `data = ...` stores the table under the name `data`, so later cells can use it.
- `.head(10)` returns the first 10 rows of the table. Colab shows the result below the cell because it is the last line of the cell.

</details>

#### 2.2 The three documents

The function `read_documents` reads the PDF report, the HTML notice and the TXT note in the same way as the app.

<details markdown="1">
<summary>Show the implementation</summary>

```python
documents = read_documents("maintenance_report_primary_school.pdf", "supplier_high_consumption_notice.html", "community_coordinator_note.txt")
```

- `read_documents(...)` needs the three file names in this order: the PDF file, the HTML file and the TXT file.
- The function extracts the text of each document, exactly as the app did in Section 1, and prints the length of each text in characters.
- `documents = ...` keeps the three texts under the name `documents`.

</details>

#### 2.3 The four statistics

The app does not send the table to the LLM. Instead, it summarises the column `electricity_consumption_kwh` with a few statistics.

<details markdown="1">
<summary>Show the implementation</summary>

```python
stats = compute_statistics(data, "electricity_consumption_kwh")
show_statistics(stats)
```

- `compute_statistics(...)` computes the mean, the median, the standard deviation, the maximum and the minimum of the column, in the same way as the app.
- `stats = ...` stores the five results under the name `stats`.
- `show_statistics(stats)` prints the five values in the format of the table of your demo report. The mean should be 6192.28.

</details>

#### 2.4 The prompt

The function `build_prompt` writes the prompt of the app from the role of the AI, the name of the column, the statistics and the documents. Read the printed prompt as the model reads it.

<details markdown="1">
<summary>Show the implementation</summary>

```python
role = "Energy Engineer"
column = "electricity_consumption_kwh"
prompt = build_prompt(role, column, stats, documents)
print(prompt)
```

- The first two lines store two pieces of text. The quotation marks tell Python that each value is text.
- `build_prompt(...)` joins the role, the four statistics, the text of the three documents and the task into one long text.
- `print(prompt)` shows that text. It is character for character the prompt that the demo app sent to Groq or Gemini in Section 1.

</details>

### Part 3 — Running the pretrained model

#### 3.1 Downloading the model

Each model on the Hugging Face Hub has a name with two parts, the organisation and the model, separated by a slash. The name `Qwen/Qwen3-0.6B` points to the model Qwen3-0.6B of the organisation Qwen.

<details markdown="1">
<summary>Show the implementation</summary>

```python
model = load_model("Qwen/Qwen3-0.6B")
```

- `load_model(...)` downloads the parameters of the model and loads them on the GPU of your session.
- The function also downloads the tokenizer, i.e., the tool that cuts text into tokens and joins tokens back into text.
- `model = ...` keeps the loaded model under the name `model`. A warning about `HF_TOKEN` may appear, and you can ignore it.

</details>

#### 3.2 Asking the model

The function `ask_model` sends the prompt to the model and shows the reply. It stops the reply after 700 tokens.

<details markdown="1">
<summary>Show the implementation</summary>

```python
answer = ask_model(model, prompt)
```

- `ask_model(model, prompt)` turns the prompt into tokens, lets the model write its reply token by token, and turns the reply back into text.
- The function shows the reply as formatted text. Then, it prints how many tokens the model read and wrote, and how long it took.
- `answer = ...` keeps the reply, so you can compare it with other replies later.

</details>

### You have finished the practical

Before you leave, discuss these three questions with your neighbour.

1. The default demo model has about 35 times more parameters than Qwen3-0.6B. Which differences in the findings can you explain with that size?
2. In Section 1, your statistics and documents travelled to the servers of Groq or Google. In Section 2, they stayed in your Colab session. When would a company prefer the second option, even with a weaker model?
3. Which of the two reports would you sign as the engineer responsible? What would you check first?

If you have time left, continue with the two advanced tasks below.

### Advanced 1 — Comparing the two models

Open your PDF report from Section 1 at section 5, "Audit Findings (AI Inference)". Then, compare it with the reply of Qwen3-0.6B. The checklist cell lists six facts that the documents and the statistics support, plus one check for invented facts.

For each fact and each model, replace `None` by `True` or `False`. A fact counts as `True` only if the findings state it **correctly**. A fact with a wrong number counts as `False`, and so does a fact that the model only implies. The last row asks the reverse question, so mark it `True` only if the findings contain no invented fact and no wrong number.

This task has no implementation on this page, because the marks depend on the replies that you received. The function `show_checklist(checklist)` draws your marks as a table and counts the facts of each model.

#### Questions for the comparison

1. Which model stated more facts correctly? Which model invented facts or numbers?
2. Both models explain why the mean exceeds the baseline of the feasibility study. Look at Figure 2 of your demo report, which shows the average of each building. Does the chart support the explanation of each model?
3. Did either model notice that the coordinator's note and the maintenance report give different baselines?

### Advanced 2 — Testing whether the model reads the documents

A useful test of any AI system removes part of the input and checks how the output changes. Without the three documents, the prompt holds only four numbers. Consequently, any mention of a fault, of a building or of a date in the reply is invented.

<details markdown="1">
<summary>Show the implementation</summary>

```python
prompt_no_documents = build_prompt(role, column, stats, None)
answer_no_documents = ask_model(model, prompt_no_documents)
```

- `None` means "nothing" in Python. Here, it tells `build_prompt` to leave out the documents.
- The prompt then states "No supporting documents were provided."
- `ask_model(...)` sends the shorter prompt to the same model. Compare its reply with the reply of 3.2.

</details>

You can run the same test on the demo model. To do so, remove the three documents in the app and generate the report again.

## When something goes wrong

### Section 1 — Demonstration

| Message | Meaning and fix |
|---|---|
| `401 Unauthorized` | The key is wrong, or it belongs to the other provider. |
| `429 Too Many Requests` | You reached the free limit of the provider. Wait one minute and try again. |
| `404` or "model not found" | The provider retired that model. Choose another model in the list. |
| A white box instead of the app | Your browser blocks the preview of Colab. Open the `gradio.live` link in a new tab instead. |
| "does not contain numeric data" | The target column holds text. Choose `electricity_consumption_kwh`. |

### Section 2 — Practice

| Message | Meaning and fix |
|---|---|
| `MISSING` in the check of 1.2 | A file is not in the Files panel, or its name differs. Upload it again with its original name. |
| `FileNotFoundError` | The same problem as `MISSING`. Check the name of the file in the Files panel, then run the cell again. |
| `NameError: name 'stats' is not defined` | You skipped a cell. Run the cells of Parts 1 to 3 again, in order. |
| `NameError: name 'true' is not defined` | You wrote `true` instead of `True`. Python needs the capital letter. |
| No GPU available | Continue on the CPU. Each answer then takes one to two minutes. |
| A warning about `HF_TOKEN` | The Hugging Face Hub suggests a personal token. The public model of this practical does not need one. |
