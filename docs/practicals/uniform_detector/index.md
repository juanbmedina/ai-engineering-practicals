---
layout: default
title: "Practical 5 — Training a small Convolutional Neural Network for a Safety Detector"
---

# Practical 5 — Training a small Convolutional Neural Network for a Safety Detector

## How this session works

**This session has two sections, and both work on the same safety problem in a simulated car workshop.** In Section 1, you run the Safety Monitor, a demonstration app that watches a video of the workshop and reports who enters a restricted zone without the right overalls. The app finds the overalls with a mask, and you calibrate that mask by hand with four sliders. In Section 2, you build the same detector in a different way. Instead of choosing the range of the mask yourself, you give the computer crops of people with and without overalls, and the computer learns the decision from those examples. At the end, you plug your model into the Safety Monitor and run it on the same scenes and the same video.

| Section | What you do | Notebook |
|---|---|---|
| 1 — Demonstration | Run the Safety Monitor and calibrate the overalls mask with the sliders | Safety Monitor |
| 2 — Practice | Train a small model that classifies the crops, then run it inside the Safety Monitor | `MEEN41490_P5_Overalls_Detector_STUDENT` |

In every practical so far, somebody else wrote the rule that the computer follows. In Practical 3, you wrote a mask by hand to find the pole in each snapshot. In Practical 4, you downloaded a pretrained language model and used it as it was. In Section 2 of this session, for the first time, you **train** a model. That is, you give the computer labelled examples, and the computer adjusts its own numbers until it separates the examples correctly.

Before you open the demonstration, read the next four sections. They describe the workshop, the detector that finds the people, the mask that decides who wears the overalls, and the workflow that joins them. Each one introduces something that you will see in the app.

## The workshop

The Safety Monitor watches a car workshop. The workshop is simulated, so no real person appears in the data. In the middle of the floor, a yellow area with two car lifts is the **restricted zone**. Only the mechanics in red overalls may enter it, because the lifts can move at any moment. Everybody else, e.g., the staff in polo shirts and jeans or the visitors, must stay outside the zone. Several cameras film the workshop from different positions, and the light changes from one camera to the next.

![A camera view of the workshop. The restricted zone around the two car lifts is outlined in red. Two workers in red overalls and one man in dark clothes stand inside it, and other people walk outside]({{ '/practicals/uniform_detector/assets/workshop_zone.jpg' | relative_url }})

The figure shows one camera view with the restricted zone outlined. Two workers in overalls stand inside the zone, which is allowed. A man in dark clothes also walks inside the zone, and that is the situation that the Safety Monitor must report. To do so, the monitor needs two pieces of information for every frame. First, where each person is. Second, whether that person wears the overalls.

## YOLO, a pretrained detector that finds the people

The Safety Monitor finds the people with **YOLO**, an object detector. The name stands for "You Only Look Once". An object detector receives a whole image and returns a list of boxes. Each box carries three things, i.e., the four corners of the box, the name of the object inside it, and a confidence between 0 and 1. The first version of YOLO appeared in 2016, and its name describes its main idea. Earlier detectors looked at thousands of candidate regions of the image one by one. YOLO passes the whole image through one network a single time, and that network predicts every box at once. For this reason, it runs fast enough for video.

**YOLO is pretrained, and you do not train it.** The people who built YOLOv8, the version used in this module, trained it on a public dataset of 80 kinds of everyday objects, such as person, car, truck, chair and dog. The weights of that training are a 22 MB file, `yolov8s.pt`, which both notebooks download once. You use the detector as it is, in the same way as you used the language model of Practical 4. The figure below shows what YOLOv8 finds in a frame of the workshop when you ask it for every kind of object it knows.

![A frame of the workshop with the boxes of YOLOv8 for every kind of object it knows, with a label and a confidence on each box]({{ '/practicals/uniform_detector/assets/yolo_many_objects.jpg' | relative_url }})

In this frame, YOLOv8 finds eight people, the two cars on the lifts, and the generator, which it calls a truck. It also labels the toolboxes as suitcases or as a bench, because its dataset has no toolbox class. A detector can only name the objects it has seen during training. For the Safety Monitor, these mistakes do not matter, because the monitor only uses one of the 80 classes.

**In this session, YOLO only looks for people.** The Safety Monitor asks the detector for the class `person` and ignores every other box. The detector then returns one box per person in the frame, under any camera and any light, because it has seen hundreds of thousands of people during its training.

**However, YOLO does not know what overalls are.** It detects people in general. For YOLO, a mechanic in red overalls and a visitor in a black T-shirt are the same thing, a `person`. Therefore, the Safety Monitor needs a second step. For every box that YOLO returns, something has to classify the person inside it as `overalls` or `no_overalls`. In the demonstration, that something is a mask. In the practice, it is the model that you train.

| Question | Who answers it | Trained by |
|---|---|---|
| Where is a person in this frame? | YOLOv8, the detector | The authors of YOLOv8, on 80 kinds of objects |
| Does this person wear the overalls? | The classifier, i.e., the mask in the demonstration and your model in the practice | You, on 250 crops, in Part 3 |

This split of the two jobs is the main design idea of the Safety Monitor. The detector answers a general question, and a general pretrained model answers it well. The classifier answers a question specific to this workshop, and no pretrained model knows the answer. Therefore, you train only the small, specific part, and you reuse the large, general part. Many engineering applications of AI follow this pattern.

## The mask that decides who wears the overalls

In the demonstration, the classifier is a mask of the kind you wrote in Practical 3. The app cuts the crop of each person out of the frame, i.e., the content of the box that YOLO found. Then, it converts the crop to the Hue, Saturation and Value (HSV) colour space, which you met in Practical 3. Four sliders select a range of hue, saturation and value. The mask keeps the pixels whose colour falls inside that range, and the app counts them. When the red pixels cover more than a set share of the crop, e.g., 15%, the person counts as `overalls`. In the demonstration, you choose the range and the share with the sliders.

![Three panels: the crop of a worker, the mask with the red pixels in white, and the crop with only the red pixels in colour. The red pixels cover 28% of the crop, so the rule says overalls]({{ '/practicals/uniform_detector/assets/mask_on_crop.png' | relative_url }})

The figure applies one range to one crop from a well-lit camera. The mask covers the overalls and little else. The red pixels make up 28% of the crop, which is above the threshold, so the rule says `overalls`. For this camera, this range gives the right answer.

**The same range under other cameras.** The figure below applies the same range and the same threshold to five workers in overalls, seen by five different cameras. From left to right, the light gets dimmer or the floor gets brighter. The overalls look darker, or more orange, and fewer of their pixels fall inside the range. From the third crop onwards, the red share drops below 15%, and the rule reports a worker as `no overalls`. In the last crop, the rule finds no red pixel at all.

![Five crops of workers in overalls under different cameras, each above its mask. The red share goes from 28% to 0%, and the rule is wrong for the last three]({{ '/practicals/uniform_detector/assets/mask_different_light.png' | relative_url }})

A wider range catches the darker overalls, and it also starts to catch the red lift, the red car and the warning stripes on the floor. A narrower range keeps the background out and loses the workers in the shadows. Every camera asks for a slightly different setting, and the calibration is a trade-off between these two effects. In the demonstration, you look for the range that works best over the whole video, and the app gives you a score for each attempt.

## The workflow of the Safety Monitor

The figure below puts the two steps together on one frame of the workshop.

![Three panels, one above the other: the camera frame, the frame with a box of YOLO around each person, and the frame with each box coloured green for overalls or red for no overalls according to the mask]({{ '/practicals/uniform_detector/assets/workflow_safety_monitor.jpg' | relative_url }})

1. **The camera frame.** The video of the workshop arrives frame by frame. The Safety Monitor also knows where the restricted zone is, because you mark its corners on the first frame.
2. **Detection with YOLO.** YOLOv8 returns one box per person, with its confidence. Every other object is ignored.
3. **Classification of each person.** The crop inside each box goes to the classifier, which answers `overalls` or `no_overalls`. The monitor draws the box in green or in red. When a red box overlaps the restricted zone, the monitor raises an alert.

In the figure, the mask with one range gives the right answer for five of the six people. The worker who crouches on the yellow floor gets a red box with 5% of red pixels, for the reason you saw in the previous section. In the demonstration, step 3 is the mask that you calibrate. In the practice, you replace step 3 with a model that you train, and steps 1 and 2 stay exactly as they are.

## Section 1 — The demonstration: calibrating the mask by hand

The Safety Monitor is a ready-made app that runs in Google Colab. You do not write any code. The notebook holds the whole app behind three cells, and the app opens inside the notebook.

> **Before you begin.** If Colab is new to you, read [Getting Started with Google Colab]({{ '/guides/getting-started-colab/' | relative_url }}) first. Download the three videos of the workshop to your computer, because you upload them to the app. Start with [`workshop_walk.mp4`]({{ '/practicals/uniform_detector/assets/data/workshop_walk.mp4' | relative_url }}){:download=""}, in which one worker in overalls walks into the restricted zone. Then, try [`workshop_walk2.mp4`]({{ '/practicals/uniform_detector/assets/data/workshop_walk2.mp4' | relative_url }}){:download=""}, a second clip of the same camera, and [`multiple_workers.mp4`]({{ '/practicals/uniform_detector/assets/data/multiple_workers.mp4' | relative_url }}){:download=""}, a clip from another camera with several workers inside the zone and other people walking past it. The notebook asks Colab for a GPU. If Colab offers none, accept the connection without GPU. The app still works, only more slowly.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12cN25jqk4h2mQBP6PRL3Jzm0VMef-hgc)

**Run the app.**

1. **Install the libraries.** Run cell 1 and wait about one minute.
2. **Load the app.** Run cell 2. This cell holds the code of the whole app, and you can double-click its title to read it.
3. **Start the app.** Run cell 3. The app appears below the cell, and a public link that ends in `gradio.live` appears above it. Click that link to open the app in a new browser tab, where it has more space.

**Calibrate the mask.**

1. Upload `workshop_walk.mp4` and click **Extract first frame**. The app runs YOLOv8 on that frame and draws a box around each person. The first time, this takes a few seconds, because the model downloads.
2. Click on the first frame to mark the corners of the restricted zone, i.e., the yellow floor. You need at least three points. This is the zone of the workflow figure above.
3. Open **How HSV segmentation works**. The diagram shows the HSV colour space, and the bright part is the region that the four sliders select. The frame next to it keeps only the pixels inside that region. Move the sliders and watch both images change together.
4. Open **Red-overall detection & HSV calibration**. Red needs two hue windows, because it sits at both ends of the hue scale. The preview marks each person on the first frame with the share of red pixels inside their box. People above the **Red pixel threshold** count as wearing the overalls. This preview is the mask of the previous sections, applied to every box of the frame.
5. Keep **Max seconds of video to process** low, e.g., 10 seconds, and click **Process video**. The annotated video and a summary appear at the end.
6. Read the calibration score in the summary. It reports the share of frames with at least one overall, the separation between the red share of the overall boxes and the other boxes, and the noise, i.e., the share of matched pixels outside the person boxes. A good calibration has a high share of frames, a large separation and a low noise.
7. Change one slider, process the video again, and compare the new row of the **Attempt history** with the previous one. Try at least three calibrations. For example, widen hue window 1 from 10 to 15, and then raise the **Minimum value (brightness)** to 40.
8. Upload `multiple_workers.mp4`, extract its first frame, mark the zone again, and process it with your best calibration. This clip comes from another camera, and several workers and other people cross the frame at the same time. Check the preview of the first frame. Does the same range count every worker, and nobody else? Clear the attempt history with the button when you change the video, because the scores of two videos are not comparable.


## Section 2 — The practice: training the overalls classifier

In Section 1, you chose the range of the mask, and the app only counted the pixels. In this section, the computer chooses. You give it 250 crops of people, labelled `overalls` or `no_overalls`, and a small Convolutional Neural Network (CNN) adjusts its own numbers until it separates the two groups. Your challenge is to design and calibrate that network until it reaches a test accuracy of 0.95 with as few parameters as possible. Then, in Part 4, you plug your best model into the workflow of the Safety Monitor in place of the mask, and you run it on the scenes and the three videos of the demonstration.

Part 0 below refreshes what you need to know about convolutional networks before you open the notebook of the practice. The notebook does not repeat it, so read it first.

## Part 0 — Convolutional networks: what you need to remember

The model that you train today is a small Convolutional Neural Network (CNN). You have seen convolutions in the lectures, and this part refreshes the four ideas that the challenge relies on. Each figure of this part comes from a real crop of the dataset. This part lives on this page only, and the notebook starts at Part 1.

### 0.1 An image is an array of numbers

A colour image is a stack of three arrays, one per colour channel (red, green and blue). Each number is the brightness of one pixel in one channel, from 0 (dark) to 255 (bright). You used the same representation in Practical 3, where you called the channels layers. The model never sees a picture. It sees these numbers.

![A 64 by 64 crop of a worker, its red, green and blue channels drawn in grey, and the 36 numbers of a 6 by 6 patch of the red channel on the torso]({{ '/practicals/uniform_detector/assets/crop_three_arrays.png' | relative_url }})

The crops of this practical have 96 rows and 96 columns. The notebook shrinks them to 64 × 64 before training, so each crop enters the model as an array of shape `(3, 64, 64)`. That is, three channels, 64 rows and 64 columns. PyTorch, the library that builds and trains the model, puts the channels first.

### 0.2 A convolution slides a small window over the image

A **convolutional layer** holds a small grid of numbers called the **kernel**, e.g., 3 × 3. The layer places the kernel on every position of the image. At each position, it multiplies the numbers under the window by the numbers of the kernel and adds the products. The result is one number. All these numbers together form a new image, the **feature map**.

![One 3 by 3 window of the red channel, the vertical-edge kernel, the resulting feature map, and the feature maps of three hand-written kernels over the whole crop]({{ '/practicals/uniform_detector/assets/convolution_kernels.png' | relative_url }})

The top row of the figure works out one position by hand. The window holds nine numbers of the red channel, and the kernel holds nine numbers of its own. The sum of the nine products is −0.19, which becomes one pixel of the feature map. The bottom row applies three hand-written kernels to the whole crop. The first kernel responds to vertical edges, the second to horizontal edges, and the third blurs the image.

The kernel decides what the feature map highlights. In a trained model, nobody writes the kernels. Training adjusts their numbers, so each kernel ends up responding to whatever pattern helps the decision. A layer with 8 kernels produces 8 feature maps. In PyTorch, the layer is `nn.Conv2d(3, 8, kernel_size=3, padding=1)`. The first argument is the number of channels that arrive, the second is the number of kernels, and `padding=1` keeps the feature map at the size of the input.

### 0.3 ReLU and max pooling

Two simple operations follow every convolution in the models of this practical.

**ReLU** keeps the positive numbers and replaces every negative number by 0. Without it, a stack of convolutions would collapse into one big convolution. With it, each layer can build patterns from the patterns of the layer before.

**Max pooling** shrinks a feature map. It divides the map into small squares, e.g., 2 × 2, and keeps only the largest number of each square. Pooling with a 2 × 2 square halves the height and the width. The map keeps the strongest responses and loses their exact position. That loss is useful, because the overalls can be anywhere in the crop.

![The ReLU curve, a feature map after ReLU, a 4 by 4 piece of it, the same piece after 2 by 2 max pooling, and the whole map after pooling]({{ '/practicals/uniform_detector/assets/relu_max_pooling.png' | relative_url }})

In the figure, the 4 × 4 piece becomes a 2 × 2 piece. Each new number is the largest of the four numbers in its quarter. In PyTorch, `nn.MaxPool2d(2)` pools with a 2 × 2 square, and `nn.MaxPool2d(8)` divides the height and the width by 8.

### 0.4 From feature maps to a decision

A model for this practical is a chain of layers. The convolutional layers and the pooling layers turn the crop into a small stack of feature maps. Then, **Flatten** lays all the numbers of that stack in one long row. A **Linear** layer multiplies each number of the row by a weight and adds the results into a single score. This is the neuron of Practical 2, applied to hundreds of inputs instead of four. Finally, the **sigmoid** maps the score to a number between 0 and 1. That number is the confidence of the model that the person wears the overalls. Values above 0.5 count as `overalls`.

![The example model drawn as a chain of boxes, from the input of 3 by 64 by 64 to a single number, with the shape of the data after each layer]({{ '/practicals/uniform_detector/assets/architecture_example.png' | relative_url }})

The notebook defines a helper, `show_architecture(model)`, which draws any model as this chain of boxes with the shape of the data after each layer. Read the shapes from left to right. The input has 3 channels of 64 × 64. The convolution with 4 kernels gives 4 maps of 64 × 64. The pooling by 8 gives 4 maps of 8 × 8. Flatten gives 4 × 8 × 8 = 256 numbers, and the linear layer turns them into 1 number. You will use this helper on your own models in Part 3, where it tells you the number that each `Linear` layer must expect.

### 0.5 How training works

**Parameters.** Every number inside the kernels and the linear layers is a parameter. The example model above has 369 of them. Training adjusts all of them, and nothing else.

**Loss.** The model gives a confidence for each training crop. The **loss** measures how far those confidences are from the labels, i.e., 1 for `overalls` and 0 for `no_overalls`. A perfect model has a loss near 0. A model that answers at random has a loss near 0.69, which is the value you will see when a model fails to learn.

**Gradient step.** Training looks at a **batch** of crops, e.g., 32, and computes the loss. Then, it works out for each parameter in which direction the loss decreases, and it moves every parameter a small step in that direction. The **learning rate** sets the size of the step. A small step is safe but slow. A large step is fast but can overshoot. One pass over the whole training set is an **epoch**.

**Training set and test set.** The model learns from the training set. The test set stays hidden during training. Therefore, the accuracy on the test set tells you how the model behaves on people it has never seen. Here the two sets come from different camera snapshots, so no person appears in both. In a real project you would also keep a third set, the validation set, to choose between models, and you would look at the test set only once at the end. To keep the session short, you compare your models on the test set directly.

**Accuracy and chance.** Accuracy is the share of correct answers. Both sets have the same number of `overalls` and `no_overalls` crops. Therefore, a model that answers at random, or always the same label, scores 0.5. That is the floor. The mask of the demonstration scores 0.68 on the test set, and that is the score to beat.

## The notebook of the practice

The notebook has four parts, and Part 3 is the challenge. Part 0 is this page, so read it before you open the notebook. Most code cells of the notebook show only a title, because their code is collapsed. Run them like any other cell, and double-click the title if you want to read the code. The code that stays visible is the code that you read or edit, i.e., the models of Parts 2.3 and 3 and the short calls of Part 4.

| Part | What you do |
|---|---|
| 1 | Load the crops and measure the mask of the demonstration on the test set |
| 2 | Meet the training loop and train a first model |
| 3 | **The challenge.** Calibrate the architecture and the training until the model reaches the target |
| 4 | Run YOLOv8 and your model together on the scenes and the three videos of the demonstration |

> **How to use this page.** The notebook holds the code, and you run its cells in order. This page explains the ideas behind each part, so that you understand what every cell does before you run it. The challenge in Part 3 has no solution on this page, because that problem is yours to solve.

> **Before you begin.** Open the notebook, save your own copy to Drive, and rename it to `MEEN41490_P5_YourName`. The practice runs on the CPU, so you do not need a GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1T2kAeJpReotKX4YmX-BWeTHGwSzImVOZ)

### The data of the session

The notebook uses the ten data files below. You do not need to download or upload them, because the setup cell of the notebook downloads them from this page into Colab and unpacks the crops. The links are here so that you can look at the data on your own computer.

- [`overalls_crops.zip`]({{ '/practicals/uniform_detector/assets/data/overalls_crops.zip' | relative_url }}){:download=""} — 354 crops of people, 96 × 96 pixels, split into a training set and a test set
- [`workshop_walk.mp4`]({{ '/practicals/uniform_detector/assets/data/workshop_walk.mp4' | relative_url }}){:download=""}, [`workshop_walk2.mp4`]({{ '/practicals/uniform_detector/assets/data/workshop_walk2.mp4' | relative_url }}){:download=""} and [`multiple_workers.mp4`]({{ '/practicals/uniform_detector/assets/data/multiple_workers.mp4' | relative_url }}){:download=""} — the three videos of Section 1
- [`scene_002.jpg`]({{ '/practicals/uniform_detector/assets/data/scene_002.jpg' | relative_url }}){:download=""}, [`scene_021.jpg`]({{ '/practicals/uniform_detector/assets/data/scene_021.jpg' | relative_url }}){:download=""}, [`scene_032.jpg`]({{ '/practicals/uniform_detector/assets/data/scene_032.jpg' | relative_url }}){:download=""}, [`scene_033.jpg`]({{ '/practicals/uniform_detector/assets/data/scene_033.jpg' | relative_url }}){:download=""}, [`scene_047.jpg`]({{ '/practicals/uniform_detector/assets/data/scene_047.jpg' | relative_url }}){:download=""}, [`scene_099.jpg`]({{ '/practicals/uniform_detector/assets/data/scene_099.jpg' | relative_url }}){:download=""} — six full camera snapshots of the workshop

Each crop of the dataset is the content of one box that YOLOv8 found in a workshop snapshot, cut as a square and resized to 96 × 96 pixels. A person labelled every crop as `overalls` or `no_overalls`.

## Part 1 — The data

### 1.1 Loading the crops

The cell reads the crops into arrays, counts them, and shows twelve random examples. The training set holds 125 crops of each class, and the test set holds 52 of each class. The notebook reports these counts as `(250, 96, 96, 3)` and `(104, 96, 96, 3)`, i.e., images, rows, columns and channels. Look at the grid. The people are small, the light changes from one crop to the next, and some crops show a person from behind. The model has to work with all of them.

![Twelve random training crops with their labels, overalls or no_overalls]({{ '/practicals/uniform_detector/assets/crops_grid.png' | relative_url }})

### 1.2 The mask of the demonstration

The cell applies the mask of the demonstration, with the same thresholds, to the 104 crops of the test set. The function `demo_colour_rule` converts a crop to HSV, counts the pixels whose hue sits in the two red windows, and says `overalls` when those pixels cover more than 15% of the crop. The result is an accuracy of 0.68. The rule misses 33 of the 52 workers in overalls, whereas it accepts none of the 52 people without overalls. That is, the rule is strict and the dark overalls fall outside its range. This number is your baseline, and every model you train in Part 3 should beat it.

## Part 2 — The training loop and a first model

### 2.1 Preparing the arrays

The model takes crops of 64 × 64 pixels, with the values scaled from 0–255 to −0.5–0.5. The cell builds the training and test **tensors**, which is the name that PyTorch gives to an array. The shape of the training tensor is `(250, 3, 64, 64)`. Compare it with the shape of Part 1. The channels moved from the last position to the second, because PyTorch expects them there.

### 2.2 The training loop

The cell defines the two functions that you call in the challenge. Its code is collapsed, because you do not need to change it. Still, you need to know what each function does, because the challenge asks you to interpret what they print.

- `train_model` runs the gradient steps of Part 0.5 for a number of epochs. After each epoch, it records the loss and the accuracies on the training set and on the test set.
- `run_experiment` trains a model from fresh random parameters, measures its test accuracy, its number of parameters and its training time, and draws the curves. Then, it adds one row to an **experiment log**. The log is a table with every model you have tried.
- `show_log`, one of the hidden helper functions, prints the log after each run. Its last column says `yes` for each run that meets the three conditions of the challenge.

Keeping such a log is how engineers tune a model without losing track. In Part 3, you will change one thing per run, and the log lets you compare every run with the ones before.

### 2.3 A first model

The cell defines a small model and trains it. The model has one convolutional layer with 4 kernels, a pooling layer and one linear layer, which gives 369 parameters. It trains for 5 epochs with a learning rate of 0.001. The notebook explains each layer of the model in one line next to the cell, i.e., the convolution of Part 0.2, the ReLU and the pooling of Part 0.3, and the Flatten and the linear layer of Part 0.4. Read those lines before you run the cell, because in Part 3 you edit exactly these layers.

![The first model drawn as a chain of boxes, with 369 parameters]({{ '/practicals/uniform_detector/assets/first_model_architecture.png' | relative_url }})

![The loss on the training set over five epochs, still falling, and the accuracies on the training set and on the test set, below the target line of 0.95]({{ '/practicals/uniform_detector/assets/first_model_curves.png' | relative_url }})

Read the output in this order. The log table gives the test accuracy, the number of parameters and the training time. The left curve shows the loss after each epoch, and the right curve shows the two accuracies against the target line at 0.95. The first model beats the mask of the demonstration, but not by much. Moreover, the loss is still going down when the training stops. The challenge starts here.

## Part 3 — The challenge: calibrate the model

Your goal is a model that reaches a **test accuracy of 0.95 or more**, with **fewer than 50,000 parameters** and **at most 60 seconds of training**. Among the models that reach the target, the one with the fewest parameters wins. This page gives no solution, because the problem is yours to solve. However, it explains what you can change and how to read the curves.

**You can change two kinds of things.** The first kind is the training, i.e., the number of epochs, the learning rate and the batch size, which are the arguments of `run_experiment`. The second kind is the architecture, i.e., the layers inside `nn.Sequential`. You can change the number of kernels of a convolution, the kernel size, the size of the pooling, the number of convolutional layers and the number of linear layers.

**The rules of shape.** Every layer must receive the shape that the layer before it produces. The table below lists the three rules that cover every edit of this practical. When you get a shape wrong, Python stops with an error that names the two numbers that did not match, and `show_architecture` shows you the shapes after each layer.

| Layer | Rule |
|---|---|
| `nn.Conv2d(in, out, kernel_size=k, padding=k // 2)` | `in` must equal the number of maps that arrive. `out` is the number of kernels, and it is yours to choose. The padding keeps the map at its size. |
| `nn.MaxPool2d(k)` | Divides the height and the width by `k`. |
| `nn.Linear(in_features, out_features)` | The first `Linear` after `Flatten` needs `in_features` equal to channels × height × width of the maps that reach it. |

**How to work.** Edit the cell of the challenge, change **one thing**, and give the run a short name that says what you changed. Every run adds a row to the log. Then, look at the curves before you decide the next change. The table below translates each shape of the curves into a next step.

| What the curves show | What it means | What to try |
|---|---|---|
| The loss is still going down at the last epoch | The training stopped too early | More epochs |
| The loss jumps up and down | The steps are too large | A smaller learning rate |
| The loss stays flat near 0.69 | The model does not learn | A larger learning rate, or a different architecture |
| Training accuracy near 1.00 and test accuracy well below | The model memorises the training crops | A smaller model, or more data |

When a run reaches the target, run it again with `seed=1`. A good configuration reaches the target with both seeds. Then, try to reach the target with fewer parameters. Training takes a few seconds for small models, so you have time for many runs.

**Pick the best model.** When you have finished, run the cell that picks the best run of your log. It keeps the model with the fewest parameters among those that reached the target, and it prints its **confusion matrix**. The confusion matrix is a 2 × 2 table. The rows are the true labels, and the columns are the answers of the model. The two diagonal cells count the correct answers. The other two cells count the two kinds of mistakes, i.e., a worker reported as an intruder and an intruder reported as a worker. The cell also shows the crops that the model got wrong, with its confidence in each one.

## Part 4 — Your model inside the Safety Monitor

Your model only knows crops. The Safety Monitor needs a decision for every person in a full camera snapshot. The cells of this part repeat the workflow of the demonstration. YOLOv8 finds the people, the notebook cuts a square crop around each person, and your best model decides. A green box means "authorised" and a red box means "not authorised". The tag under each box shows what the mask of the demonstration says with its default range.

The whole workflow fits in a few lines, and the notebook holds them in the function `annotate`. That function sits in the collapsed cell that opens Part 4, together with its helpers and the line that loads YOLOv8. The lines that matter are these.

```python
from ultralytics import YOLO
detector = YOLO("yolov8s.pt")
result = detector.predict(bgr, classes=[0], conf=0.35, imgsz=1280, verbose=False)[0]
for b in result.boxes:
    box = [int(v) for v in b.xyxy[0].tolist()]
    prob, rule = classify_crop(square_crop(bgr, box))
```

- `YOLO("yolov8s.pt")` loads the pretrained detector. The weights download the first time. The letter `s` stands for "small", because YOLOv8 comes in five sizes from `n` (nano) to `x` (extra large). This one has about 11 million parameters, whereas your model has fewer than 50,000.
- `classes=[0]` keeps only the boxes of class 0, which is `person` in the dataset of YOLOv8. The monitor ignores the cars and the toolboxes.
- `conf=0.35` keeps only the boxes with a confidence of at least 0.35. A lower value finds more people and also more false boxes.
- `imgsz=1280` resizes the frame so that its longest side has 1280 pixels before detection. A larger size helps the detector to find the small people far from the camera, and it costs more time.
- `b.xyxy[0]` holds the four corners of one box, i.e., the column and the row of the top-left corner and of the bottom-right corner.
- `square_crop` cuts a square around the box with a margin of 15%, and `classify_crop` resizes the square to 96 × 96 and asks your model for its confidence.

### 4.1 Camera snapshots

The cell runs the pipeline on two snapshots of the workshop. Compare the colour of each box with the tag under it. The figure below shows the result with a model that reached the target. In this snapshot, the mask reports several workers in overalls as `no overalls`, whereas the model draws a green box around them with a confidence near 1.00.

![A workshop snapshot with a box around each person. Green boxes mark the workers that the model accepts, red boxes mark the other people, and a tag under each box gives the answer of the mask]({{ '/practicals/uniform_detector/assets/scene_021_annotated.png' | relative_url }})

### 4.2 The video from the demonstration

The cell runs the same pipeline on the first video of the demonstration through the function `annotate_video`. It processes one frame out of every three, shows six of them, and writes the annotated video to `workshop_walk_annotated.mp4`. You can download that file from the Files panel. Watch the worker who walks into the yellow zone. The model keeps a green box on him from frame to frame, whereas the mask changes its answer with the light.

![Six frames of the annotated video. A worker in red overalls walks into the yellow zone, and the model keeps a green box on him]({{ '/practicals/uniform_detector/assets/video_frames.png' | relative_url }})

### 4.3 Your model on the three videos

In Section 1, you calibrated the mask on three videos, and your model deserves the same test. The cell runs `annotate_video` on `workshop_walk2.mp4` and on `multiple_workers.mp4`. The first clip comes from the same camera as the video of 4.2. The second clip comes from another camera, with several workers inside the zone and other people walking past it. Each video takes about a minute on the CPU, and the annotated copies appear in the Files panel.

Look at the frames with your team. Does the model keep the same answer for one person from frame to frame? Does it accept every worker in overalls, and nobody else? Which video gives it the most trouble, and why? A model that works on the first camera and fails on the second has seen too few crops from the second camera. If that happens, go back to Part 3 and improve the model. Then, run the cell that picks the best model and run the videos again.

## You have finished the practical

**Well done! You have finished every required task of Practical 5.** Compare the boxes of Part 4 with the calibration you wrote down at the end of Section 1. Then, discuss these questions with your team before you leave.

1. Look at your log. Which change gave the largest gain in test accuracy? Which change cost the most parameters for the smallest gain?
2. Which errors matter more for a safety monitor, a missed intruder or a false alarm on a worker? Look at the confusion matrix of your best model.
3. You compared your models on the test set and then reported the accuracy of the best one on the same test set. Why is that number optimistic? What would you change in a real project?
4. The crops come from one simulated workshop. What would you expect on a video from a different building, with a different floor colour? What data would you need to collect?
5. The overalls could change colour next year. Compare the effort of updating the mask with the effort of updating your model.

Question 4 matters for your team project. Your data collection plan described where your data would come from. This practical shows what happens when a rule tuned on one source meets another source, and what a trained model needs instead, i.e., labelled examples from every source it will meet.

## Beyond the practical — Optional advanced levels

If you finish before the end of the session, the notebook continues with three optional cells. Each one looks inside the model from a different angle.

**Advanced 1 — What the first layer learned.** The cell shows the kernels of the first convolutional layer of your best model, and the feature maps they produce on one crop. Compare them with the hand-written kernels of Part 0.2. Nobody wrote these kernels. Training found them.

![The kernels of the first layer of a trained model, drawn as small colour squares, and the feature map that each one produces on a test crop]({{ '/practicals/uniform_detector/assets/first_layer_kernels.png' | relative_url }})

**Advanced 2 — A model that sees one pixel at a time.** A convolution with `kernel_size=1` looks at a single pixel, so it can only combine the three colour values of that pixel. Such a model is a learned colour rule. Train one and compare it with your best model. Then, think about what a 3 × 3 kernel can see that a 1 × 1 kernel cannot.

**Advanced 3 — Less data.** The cell trains your best architecture on only 40 crops. The training accuracy climbs towards 1.00, whereas the test accuracy stays behind. This gap is called **overfitting**. The model memorises the 40 crops instead of learning the overalls, and the gap grows with the size of the model.

![The loss and the accuracies of the best architecture trained on 40 crops. The training accuracy reaches 1.00 and the test accuracy stays well below it]({{ '/practicals/uniform_detector/assets/overfitting_40_crops.png' | relative_url }})

## Something went wrong?

- **The setup cell cannot download the data files.** Run the setup cell again. If the error repeats, download the ten files listed under "The data of the session", upload them in the Files panel, and run the setup cell once more.
- **`RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x1024 and 256x1)`.** The first `Linear` layer expects 256 numbers but receives 1024. Set its first argument to the number that `show_architecture` prints before `Linear`.
- **`RuntimeError: ... expected input[32, 8, 32, 32] to have 4 channels`.** A `Conv2d` expects 4 maps but receives 8. Set its first argument to the number of kernels of the convolution before it.
- **Training is slow.** Reduce the number of kernels or add pooling earlier. Models under 50,000 parameters train in well under a minute on the CPU.
- **Part 4 shows no boxes.** Run the first cell of Part 4 again, so that the weights of YOLOv8 download.

## Quick reference of PyTorch commands

| Python | Meaning |
|---|---|
| `nn.Sequential(a, b, c)` | A model that applies the layers `a`, `b` and `c` in order |
| `nn.Conv2d(in, out, kernel_size=3, padding=1)` | A convolution with `out` kernels of 3 × 3 over `in` channels, which keeps the size of the map |
| `nn.ReLU()` | Replaces every negative number by 0 |
| `nn.MaxPool2d(k)` | Keeps the largest number of each `k` × `k` square, which divides the height and the width by `k` |
| `nn.Flatten()` | Lays all the numbers of the maps in one row |
| `nn.Linear(in, out)` | A layer of `out` neurons, each with `in` weights and a bias |
| `show_architecture(model)` | Draws the model and the shape of the data after each layer |
| `run_experiment(name, model, epochs, learning_rate, batch_size, seed)` | Trains the model, draws the curves and adds a row to the log |
| `show_log()` | Prints the log of every run |
| `YOLO("yolov8s.pt")` | Loads the pretrained detector |
| `detector.predict(image, classes=[0], conf=0.35)` | Finds every person in the image with a confidence of at least 0.35 |
