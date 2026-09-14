---
layout: default
title: "Getting Started with Google Colab"
---

# Getting Started with Google Colab

Google Colab is a document that runs code. A conventional document holds paragraphs, whereas a notebook holds cells, and every cell is either a text cell or a code cell. Text cells hold notes formatted with Markdown, and code cells hold Python and display their output directly underneath.

Three facts about that arrangement matter today.

**The code does not run on your laptop.** Your browser sends each cell to a virtual machine (VM) in a Google data centre, and that machine executes the code and returns the result to your screen.

**The machine is temporary.** Google lends you a VM for the length of one session, and it takes the machine back when you close the tab or leave the notebook idle. Your notebook is stored permanently in Google Drive, however everything held in the machine's memory disappears with the session.

**You can request a faster machine.** The default processor is a CPU, which is sufficient for every practical in this module. However, when an exercise involves training instead of loading a finished model, you could switch to a GPU.

## Take your own copy of the notebook

Most practicals start from a shared notebook instead of a blank one. That file does not belong to you, and Colab makes the limitation obvious the moment you open it, before you type anything.

1. Open the link shared in class or on Brightspace. The notebook opens in **view mode** — a banner across the top reads something like *"You do not have permission to save changes to this file"*, and the **Copy to Drive** icon appears in the toolbar instead of a save icon.
![A read-only banner shown when opening a shared Colab notebook you do not own](/guides/getting-started-colab/assets/p16.png)

2. Go to **File → Save a copy in Drive**, or click the **Copy to Drive** toolbar icon if it is shown.
![The File menu with Save a copy in Drive highlighted](/guides/getting-started-colab/assets/p17.png)

3. Colab creates a file named `Copy of <original name>.ipynb` inside your own **Colab Notebooks** folder and opens it in a new tab. That copy belongs to you, and it saves itself automatically as you type.
![The copied notebook, now owned by the student in their own Drive](/guides/getting-started-colab/assets/p18.png)

4. Click the title and rename the copy to `MEEN41490_P2_YourName`. Then close the original read-only tab, because you will not need it again.

Your copy now lives in your Google Drive account. Therefore, you can reopen it from any computer, share it with a teammate, and recover an earlier version through **File → Revision history**. The entry **File → Locate in Drive** shows you where the file sits.
![The File menu with Locate in Drive highlighted](/guides/getting-started-colab/assets/p15.png)

## Find these four things

**The Runtime menu** — controls the machine your code runs on, and the only menu you need today.
![The Runtime menu in Colab](/guides/getting-started-colab/assets/p21.png)

**+ Code and + Text, top left** — insert a new cell below the selected one.
![The + Code and + Text buttons at the top left of a Colab notebook](/guides/getting-started-colab/assets/p22.png)

**Top right corner** — connection status, showing *Connect* before you start and RAM and disk usage afterwards.
![The connection status indicator in the top right corner of Colab](/guides/getting-started-colab/assets/p23.png)

**Left sidebar, folder icon** — **Files**, the temporary storage of the virtual machine.
![The Files panel opened from the folder icon in the left sidebar of Colab](/guides/getting-started-colab/assets/p24.png)

Two keyboard shortcuts save time. **Shift + Enter** runs the current cell and moves the selection to the next one, whereas **Ctrl + Enter**, or **Cmd + Enter** on Mac, runs the current cell and leaves the selection in place.

## Run the setup cell

Run the first code cell of the notebook now, before you read any further. It installs the simulator, and it defines the helper functions used later in the session. The first run takes about thirty seconds, because Colab allocates your virtual machine at that moment and downloads the simulator. Every later run is immediate.

The cell prints `Ready.` when it finishes.

## One rule about order

All cells in a notebook share one memory, and they fill that memory in the order in which you run them, not in the order in which they appear on the screen. Editing the text of a cell changes nothing by itself, because code takes effect only at the moment you execute it.

> **Watch the number in square brackets beside each code cell.** That counter records the order in which the cells actually ran. `[*]` means that a cell is still running, and empty brackets mean that a cell has never run in this session. When a notebook behaves strangely, that column usually explains the reason.

You will meet this behaviour today as an error, and the error looks like the following.

```
NameError: name 'controller' is not defined
```

Read that line as an ordinary sentence. You used a name that the machine was never given. The cure is **Runtime → Run all**, which rebuilds the whole notebook from the top.

| Runtime menu option | Use it when |
|---|---|
| **Run all** | You reopen a notebook, or the results stop making sense. The safe default. |
| **Run before** | You are fixing one cell and need everything above it loaded into memory |
| **Run cell and below** | You changed an early cell and want to refresh everything downstream of it |
| **Interrupt execution** | A cell is stuck and you want it to stop without clearing the memory |

Finally, open **Runtime → Change runtime type**, look at the available options, and leave the setting on **CPU** for today.
![The runtime type dialog in Colab, with CPU selected](/guides/getting-started-colab/assets/p61.png)

A GPU contains thousands of small cores that perform simple arithmetic in parallel, and neural networks consist almost entirely of that kind of arithmetic. Consequently, training that needs an hour on a CPU can finish in a minute on a GPU. Free GPU access is rationed across all Colab users, so request one only for a task that genuinely needs it. Today's neuron has four numbers in it, and a CPU handles it instantly.
