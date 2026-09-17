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

## Open a notebook

1. Open **[colab.research.google.com](https://colab.research.google.com/)** and sign in with a Google account.
2. In the *Open notebook* window, click **New notebook** at the top left.
![The Open notebook window in Colab, with the New notebook button highlighted at the top left]({{ '/guides/getting-started-colab/assets/p13.png' | relative_url }})
3. Click the title `Untitled0.ipynb` and rename the file to `MEEN41490_P2_YourName`.
![Renaming the notebook title in Colab]({{ '/guides/getting-started-colab/assets/p14.png' | relative_url }})

Your notebook now lives in your Google Drive account, and it saves itself automatically as you type. Therefore, you can reopen it from any computer, share it with a teammate, and recover an earlier version through **File → Revision history**. The entry **File → Locate in Drive** shows you where the file sits.
![The File menu with Locate in Drive highlighted]({{ '/guides/getting-started-colab/assets/p15.png' | relative_url }})

## Find these four things

**The Runtime menu** — controls the machine your code runs on, and the only menu you need today.
![The Runtime menu in Colab]({{ '/guides/getting-started-colab/assets/p21.png' | relative_url }})

**+ Code and + Text, top left** — insert a new cell below the selected one.
![The + Code and + Text buttons at the top left of a Colab notebook]({{ '/guides/getting-started-colab/assets/p22.png' | relative_url }})

**Top right corner** — connection status, showing *Connect* before you start and RAM and disk usage afterwards.
![The connection status indicator in the top right corner of Colab]({{ '/guides/getting-started-colab/assets/p23.png' | relative_url }})

**Left sidebar, folder icon** — **Files**, the temporary storage of the virtual machine.
![The Files panel opened from the folder icon in the left sidebar of Colab]({{ '/guides/getting-started-colab/assets/p24.png' | relative_url }})

Two keyboard shortcuts save time. **Shift + Enter** runs the current cell and moves the selection to the next one, whereas **Ctrl + Enter**, or **Cmd + Enter** on Mac, runs the current cell and leaves the selection in place.

## Try it: run a cell yourself

Before moving on, run one small cell of your own. Click **+ Code** to add a new cell and type the following two lines. Then run the cell with **Shift + Enter**.

```python
wheels = 4
print("Total wheels on the robot:", wheels * 2)
```

The number in square brackets to the left of the cell changes from empty to a number, and the printed line appears directly underneath. That number and that output are the two things to check every time you run a cell.

Now edit the cell so that `wheels = 6` instead of `4`, and run it again. The printed result updates, because you executed the change; simply typing the new number would not have been enough.


| Runtime menu option | Use it when |
|---|---|
| **Run all** | You reopen a notebook, or the results stop making sense. The safe default. |
| **Run before** | You are fixing one cell and need everything above it loaded into memory |
| **Run cell and below** | You changed an early cell and want to refresh everything downstream of it |
| **Interrupt execution** | A cell is stuck and you want it to stop without clearing the memory |

Finally, open **Runtime → Change runtime type**, look at the available options, and leave the setting on **CPU** for today.
![The runtime type dialog in Colab, with CPU selected]({{ '/guides/getting-started-colab/assets/p61.png' | relative_url }})

A GPU contains thousands of small cores that perform simple arithmetic in parallel, and neural networks consist almost entirely of that kind of arithmetic. Consequently, training that needs an hour on a CPU can finish in a minute on a GPU. Free GPU access is rationed across all Colab users, so request one only for a task that genuinely needs it. Today's neuron has four numbers in it, and a CPU handles it instantly.

## Take your own copy of the notebook

You will use this the first time your instructor shares a link to a practical notebook, instead of creating a new one from scratch as above.

Most practicals start from a shared notebook instead of a blank one. That file does not belong to you, and Colab makes the limitation obvious the moment you open it, before you type anything.

1. Open the notebook link shared in class or on Brightspace. The notebook opens in **view mode** — a banner across the top reads something like *"Changes will not be saved"*.
![A read-only banner shown when opening a shared Colab notebook you do not own]({{ '/guides/getting-started-colab/assets/p16.png' | relative_url }})

2. If you click in that button, a window going to show you the **Copy to Drive** option. Select that option to own a copy of the notebook on your drive.

![A read-only banner shown when opening a shared Colab notebook you do not own]({{ '/guides/getting-started-colab/assets/p17.png' | relative_url }})

3. Click the title and rename the copy to `MEEN41490_P2_YourName`. Then close the original read-only tab, because you will not need it again.

Your copy now lives in your Google Drive account, exactly like a notebook created from scratch, and everything from the section above (autosave, revision history, Locate in Drive) applies to it in the same way.