---
layout: default
title: "Demonstration — Algovivo Lab for Soft-Body Locomotion"
---

# Demonstration — Algovivo Lab for Soft-Body Locomotion

## Workflow of the app

In this demonstration, you design a small soft-body creature with sliders, and an evolutionary search finds a muscle pattern that moves it along the floor. The app runs in Google Colab, so you need no code, no API key and no GPU. The physics simulation and the search run in your own web browser, with the open-source library [algovivo](https://github.com/juniorrojas/algovivo) by Junior Rojas.

| Section | Content | Update |
|---|---|---|
| 1. Inputs | Sliders for the shape and the training budget, and a still preview of the shape | The preview, after every slider move |
| 2. Processing | The creature as data: points, triangles, muscles and training settings | At each click on **Train** |
| 3. Outputs | The simulation, the progress of the training and a replay of the best muscle pattern | At each click on **Train** |

## Creature model and evolutionary search

The creature is a soft body made of points, triangles and muscles. Each muscle joins two points and pulls them together when it contracts. The spine is a row of points at the head height, and the feet are points on the floor between minus and plus the leg spread. Each triangle joins one segment of the spine to one foot and resists any change of its shape, so the body behaves as an elastic solid.

A muscle pattern describes the contraction of every muscle over time as a wave, with one frequency for the whole creature and one amplitude and one phase for each muscle. The score of a pattern is the distance that the centre of the creature travels along the x axis during one trial. Each physics step covers 0.033 s of simulated time, so the default trial of 100 steps lasts 3.3 s.

## Preparation before the session

1. **Browser.** The app downloads algovivo from `cdn.jsdelivr.net` when it starts. Some networks and ad blockers block that site, so use an open network or turn off the ad blocker for the app.
2. **Runtime.** The default CPU runtime of Colab is enough, because the simulation runs in your browser and not in Colab.
3. **Colab.** If Colab is new to you, read [Getting Started with Google Colab]({{ '/guides/getting-started-colab/' | relative_url }}) first.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mw0gFpTE-JjK-sjSVzAG4MWnkIL0QIA3)

## Starting the app

1. Run cell 1 to install the libraries, which takes about 20 seconds. If the cell asks you to restart, select **Runtime > Restart session** and go straight to cell 2.
2. Run cell 2 to load the app.
3. Run cell 3. The app appears below the cell, and a link that ends in `gradio.live` appears above it. Open that link in a new browser tab if the app looks too small.

## Steps in the app

1. **Shape.** Move the five sliders under **Shape & Structure** and watch the **Creature preview** next to them. The blue marks show the body length, the height and the leg spread, and the line below the preview counts the points, triangles and muscles.
2. **Training budget.** Set the **Population size**, the number of **Generations** and the **Physics steps per trial**. A larger budget gives the search more chances to find a good pattern, at the cost of a longer training.
3. **Training.** Click **Train**, and the panel at the top of the viewer shows the current generation and the best distance so far. The app also trains the default creature once when it opens.
4. **Replay.** After the training, the viewer replays the best pattern in a loop. Each replay lasts as many steps as one trial, so its final distance matches the training score. Red muscles contract, and pink muscles are relaxed.
5. **Data.** The **Processing** box lists the coordinates of the points (`pos`), the triangles and the muscles of the creature that you trained. Scroll inside the box to read the whole list.


## Common errors and solutions

- **White box instead of the app.** Open the `gradio.live` link in a new browser tab.
- **Viewer stuck on "Loading algovivo...".** The browser cannot reach `cdn.jsdelivr.net`. Use another network or turn off the ad blocker for the app, and click **Train** again.
- **Creature that falls flat.** A creature with one spine node has no triangles, so it cannot keep its shape. Add spine nodes.
- **Slow training.** Large populations, many generations and many steps slow down the training, because it runs in your browser.
- **Expired link.** The public link works only as long as your notebook runs, and anyone with the link can open your app.
