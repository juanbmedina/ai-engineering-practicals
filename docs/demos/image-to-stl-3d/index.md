---
layout: default
title: "Demonstration — Image to STL Studio"
---

# Demonstration — Image to STL Studio

## Workflow of the app

In this demonstration, you turn a photo of a small object into a 3D-printable STL file, add a part to it from a text prompt, and test its strength. The app, called 3D Model Studio, runs entirely in Google Colab, so you need no code and no API key.

| Tab | Tool | Input | Output |
|---|---|---|---|
| 1. Image to STL | TripoSR, a pretrained image-to-3D model | A photo of one object | An `.stl` file of the object |
| 2. AI Editing | Shap-E, a generative text-to-3D model | An `.stl` file, a point and a prompt, e.g., "a handle" | The model with the new part fused to it |
| 3. Structural Analysis | Finite element analysis (FEA), without AI | An `.stl` file, a support, a load and a material | A safety factor and a stress map |

## Tools inside the app

**TripoSR** reconstructs a 3D mesh from a single photo, after a second pretrained model removes the background. The model infers the hidden side of the object from similar objects in its training data, so the result is plausible but never a measurement. A higher **Mesh resolution** gives finer detail at the cost of more time and more GPU memory.

**Shap-E** generates a new part from your prompt, starting from random noise, so the same prompt gives a different part every time. The app then scales the part, places it at the point that you click, and fuses it with your model into one closed solid.

**FEA** fills the model with small tetrahedra, fixes the support point, applies the force at the load point, and computes the stress in every element. The app then reports a safety factor, defined as the strength of the material divided by the 99th percentile of the stress. A value of 1.5 or more means SAFE, a value from 1.0 to 1.5 means RISK, and a value below 1.0 means HIGH FRACTURE RISK. The three materials, PLA, PETG and ABS, have strengths of 50, 48 and 40 MPa in the app. The analysis treats the plastic as uniform in every direction, whereas a real printed part is weaker between its layers.

## Preparation before the session

1. **GPU.** The notebook requests a free T4 GPU, which Tabs 1 and 2 need, whereas Tab 3 also runs on the CPU. You can check it under **Runtime > Change runtime type**.
2. **Photo.** Use one small object, fully visible and well lit, on a plain background, e.g., a mug, a toy or a shoe.
3. **Colab.** If Colab is new to you, read [Getting Started with Google Colab]({{ '/guides/getting-started-colab/' | relative_url }}) first.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1l3ThV4IePMXWBJJViesdkMbdTFR4RI2G)

## Starting the app

1. Run cell 1 to install the libraries, which takes about two minutes. If the cell asks you to restart, select **Runtime > Restart session** and go straight to cell 2.
2. Run cell 2 to load the app and download the models, which takes three to five minutes once per session.
3. Run cell 3, and open the `gradio.live` link that appears above the app in a new browser tab.

## Steps in each tab

1. **Image to STL.** Upload your photo, set the **Target size** in millimetres, keep the **Mesh resolution** at 128, and click **Generate STL**. Then, download the `.stl` file.
2. **AI Editing.** Upload the `.stl` file and click **Load and show views**. Click a point on one of the views, type a prompt, and click **Add part with AI**.
3. **Structural Analysis.** Upload an `.stl` file and click **Load views**. Mark the **Fixed point (support)** and the **Load point (force)** on the views, choose the material and the force, and click **Analyze structure**. In the stress maps, red marks the region most likely to fail.

## Common errors and solutions

- **White box instead of the app.** Open the `gradio.live` link in a new browser tab.
- **Warnings from pip in cell 1.** These are usually harmless if the cell ends with "Done".
- **`CUDA out of memory`.** Lower the mesh resolution, or select **Runtime > Restart session** and run cells 2 and 3 again.
- **Background removal cuts the object.** Untick **Remove background automatically (AI)** and use a photo with a plain background.
- **Failure of the structural analysis.** Generate the model again in Tab 1 with a lower mesh resolution.
- **Expired link.** The public link works only as long as your notebook runs, and anyone with the link can open your app.
