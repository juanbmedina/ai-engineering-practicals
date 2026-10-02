---
layout: default
title: "Practical 3 — Data manipulation with images"
---

# Practical 3 — Data manipulation with images

## What you will do today

In Practical 2, your neuron balanced the pole with four measurements that the simulator handed over, already clean and in the right order. Real systems rarely receive data in that form. A sensor delivers raw data, and somebody has to turn it into the few numbers that a model can use.

Today the sensor is a camera, and its raw data is a photo, 50 times per second. For the computer, a photo is a large table of numbers that also happens to be visible on the screen. Consequently, every time you select, filter or average those numbers, you can look at the result and check it with your own eyes.

You will write the Python code that turns each photo into the four measurements of Practical 2. Then the neuron will balance the pole using only what your code extracts, and you will adapt that code to a cart of your own.

![A cart balancing a pole, controlled by a neuron that only sees camera photos. Red squares mark the pivot and the pole centre found in each photo]({{ '/practicals/inverted_pendulum_camera_vision/assets/camera_balanced.gif' | relative_url }})

The red squares in the animation mark what the camera found in each photo: the pivot and the centre of the pole. Everything the neuron knows comes from those two points.

> **How to use this page.** Work in the notebook. Every cell marked `# YOUR CODE HERE` is yours to fill, and the notebook tells you what to write. **Try it yourself first.** If you are stuck, come back to the matching section of this page and click **Show the implementation**, which reveals the code and explains it line by line. The challenge in 4.2 and the advanced levels have no implementation here. Those are yours to solve.

> **Before you begin.** Open the notebook, save your own copy to Drive, and rename it to `MEEN41490_P3_YourName`. If you need a reminder, see [Getting Started with Google Colab]({{ '/guides/getting-started-colab/' | relative_url }}).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1RUWvuuhKjnhqFDpl_Bvi_g8Wbj_w9j8J/view?usp=sharing)

## Part 0 — How a computer sees a photo

Read this part before you start. It takes five minutes, and every cell of the notebook uses it.

**A photo is a grid of pixels.** A pixel is the smallest square of a photo, and it has a single colour. The photos in this practical have 400 rows and 600 columns of pixels, that is, 240 000 small squares. Zoom far enough into any photo and the squares appear. For the computer, each square is a few numbers in a table, and the picture on the screen is only one way of displaying that table.

![A photo of the practice cart, and a zoom on a 12 by 12 square of it where the individual pixels are visible]({{ '/practicals/inverted_pendulum_camera_vision/assets/pixels_zoom.png' | relative_url }})

**A position is a row and a column.** Rows count from the top of the photo **downwards**, and columns count from the left to the right. Both start at zero. Python always takes the row first: `image[230, 305]` is the pixel in row 230, column 305.

**A colour is three numbers.** Each pixel stores how much red, green and blue light it contains, each from 0 (none) to 255 (full). Mixing the three gives every colour on the screen.

| Pixel | Colour |
|---|---|
| `[255, 0, 0]` | red |
| `[0, 0, 255]` | blue |
| `[255, 255, 255]` | white |
| `[0, 0, 0]` | black |

Consequently, a colour photo is three grids of numbers stacked together, one per colour, and its size is written `(400, 600, 3)`. Each grid is called a layer. Below, each layer of the practice photo is drawn in grey: white where that colour is at full strength, black where it is absent. The white background is bright in all three layers. The red pole is bright only in the red layer, where it blends into the background, and it appears black in the other two.

![The red, green and blue layers of the practice photo, each drawn in grey]({{ '/practicals/inverted_pendulum_camera_vision/assets/rgb_layers.png' | relative_url }})

**Finding an object is asking every pixel a question.** The camera software you write today does two things. First, it asks every pixel *are you the colour of the pole?* The answers form a new picture of `True` and `False` values, called a **mask**. Second, it averages the rows and the columns of the `True` pixels. The result is the **centre** of the pole, in the same way as a centre of mass.

![The mask of the pole: only the pixels with the colour of the pole are black]({{ '/practicals/inverted_pendulum_camera_vision/assets/pole_mask.png' | relative_url }})

With the centre of the pole and the position of the pivot, one line of trigonometry gives the pole angle. Two consecutive photos give the velocities. That is the whole camera.

None of these operations is specific to photos. Selecting values, asking every value a question, counting the answers and averaging the values that pass are the same steps you would apply to a log of temperature readings or to a spreadsheet of test results. The photo only makes each step visible.

This is all the knowledge you need for today. Open the notebook with the **Open in Colab** button at the top of this page, and work through it from the setup cell onwards. Each section of this page matches one section of the notebook. Whenever you are stuck, come back to the matching section here and open **Show the implementation**.

## Part 1 — Setup

Type the **last two digits of your student number** in the box of the setup cell, for example `37`, and run it. It prints `Ready.` when it finishes. Your two digits choose the colours of your own cart, which you meet in Part 4. Your neighbour's cart will probably look different.

## Part 2 — From numbers to pictures

The four cells of this part introduce the operations that the camera needs, on datasets small enough to check by hand. Each operation works in the same way on four numbers and on 240 000 pixels.

### 2.1 A vector, and the neuron in one line

A vector is a row of numbers, and `np.array(...)` stores one. It behaves like the lists of Practical 2, with one extra ability. Arithmetic acts on all its numbers at once. The cell asks you to compute the neuron of Practical 2 with it.

<details markdown="1">
<summary>Show the implementation</summary>

```python
measurements = np.array([0.10, -0.30, 0.05, 0.40])
total = np.sum(weights * measurements) + bias
print("Total:", round(total, 2))
```

- `np.array([...])` turns the list of four measurements into an array.
- `weights * measurements` multiplies the two arrays element by element, that is, the first weight by the first measurement, the second by the second, and so on. `np.sum(...)` adds the four products, and `+ bias` adds the bias. This is the neuron of Practical 2 in a single line.
- `round(total, 2)` keeps two decimals. Without it Python may print `0.31000000000000005`, which is the same number. Computers store decimals with a tiny rounding error.

</details>

### 2.2 Ask every number a question

`measurements > 0` compares every number with zero at once, and answers `True` or `False` for each one. Python counts `True` as 1 and `False` as 0, so the sum of the answers is the number of `True` answers. This operation is called a filter, and it is the most common operation on any dataset. The camera uses it to find the pole.

<details markdown="1">
<summary>Show the implementation</summary>

```python
print(measurements > 0)
print(np.sum(measurements > 0))
```

The first line prints `[ True False  True  True]`, one answer per measurement. The second line adds the answers, 1 + 0 + 1 + 1, and prints `3`.

</details>

### 2.3 A matrix

A matrix is a grid of numbers. Two kinds of brackets appear in this cell, and they mean different things.

| Brackets | Meaning | Example |
|---|---|---|
| square `[ ]` | read a position inside a variable | `grid[0, 1]` is the number in row 0, column 1 |
| round `( )` | run a function on something | `print(grid)`, `np.sum(grid)` |

`grid.shape` takes no brackets, because it is not a function. It is information that the matrix carries with it: its size.

<details markdown="1">
<summary>Show the implementation</summary>

```python
print(grid.shape)
print(grid[1, 2])
```

- `grid.shape` is `(3, 5)`: 3 rows and 5 columns.
- `grid[1, 2]` reads row 1, column 2. Both count from zero, so this is the second row and the third column, which holds a `1`.

</details>

### 2.4 A matrix is a picture

Paint every 0 white and every 1 black, and the matrix becomes a picture in which each number is a pixel. The 1s of `canvas` draw a small vertical pole.

![A 7 by 7 matrix drawn as a picture: the 1s form a vertical pole in column 3]({{ '/practicals/inverted_pendulum_camera_vision/assets/canvas.png' | relative_url }})

The line `rows, cols = np.nonzero(canvas)` is already in the notebook. `np.nonzero` finds every pixel that is not zero and returns **two** lists: the rows of those pixels and their columns. Writing two names before the `=` stores the first list in `rows` and the second in `cols`. Here `rows` is `[1 2 3 4 5]` and `cols` is `[3 3 3 3 3]`. Your task is the average of each list. The canvas is small enough to check by eye, so you can confirm the method before you trust it on a real photo.

<details markdown="1">
<summary>Show the implementation</summary>

```python
print("Centre:", rows.mean(), cols.mean())
```

`.mean()` computes the average of a list. The average row is (1 + 2 + 3 + 4 + 5) / 5 = 3, and every column is 3, so the centre of the pole is row 3, column 3. That position matches the middle of the bar in the picture.

</details>

## Part 3 — The practice camera

You build your camera on a practice cart, painted in simple colours: a red pole and a blue axle. The axle is the pivot, that is, the hinge between the pole and the cart. `take_photo()` returns one photo and the four true measurements at that instant, so you can check your camera against the truth. Comparing a result with a known answer is the simplest way to validate any data processing.

### 3.1 Look at the photo

The cell asks you to print the size of the photo, draw it, and read two pixels.

<details markdown="1">
<summary>Show the implementation</summary>

```python
print(image.shape)
show_matrix(image)
print(image[0, 0])
print(image[230, 305])
```

- `image.shape` is `(400, 600, 3)`: 400 rows, 600 columns and 3 colour layers.
- `show_matrix(image)` draws the photo with its row and column axes. You will use those axes in Part 4.
- `image[0, 0]` is the top-left pixel, which belongs to the white background: `[255 255 255]`.
- `image[230, 305]` falls inside the pole: `[255 0 0]`, pure red.

![The practice photo with its row and column axes]({{ '/practicals/inverted_pendulum_camera_vision/assets/practice_photo.png' | relative_url }})

</details>

### 3.2 Find every pole pixel

`red = image[:, :, 0]` picks the whole red layer. The colon `:` means *all of them*, so the expression reads *all rows, all columns, layer 0*. The notebook does the same for green and blue.

A pixel belongs to the pole only when its red **and** its green **and** its blue match the pole colour. The notebook gives you the line with green and blue already joined by `&`, which is the symbol for *and*. Run it as it is first. The mask then also keeps the black cart and the rail, because black has no green and no blue either. The red condition is missing.

<details markdown="1">
<summary>Show the implementation</summary>

```python
pole_mask = (green == POLE_COLOUR[1]) & (blue == POLE_COLOUR[2]) & (red == POLE_COLOUR[0])
```

- `green == POLE_COLOUR[1]` asks every pixel of the green layer *is your green equal to 0?* The answer is a grid of `True` and `False`. The blue condition asks the same about blue.
- `red == POLE_COLOUR[0]` asks every pixel *is your red equal to 255?* The black cart fails this question, so the new condition removes it from the mask.
- `&` keeps a pixel `True` only when all three answers are `True`. Each comparison needs its own round brackets, otherwise Python mixes up the order of the operations.

The mask drops from 3276 to 1289 `True` pixels, which is the area of the pole in pixels. The picture of the mask is the result of the filter, so you can see at once whether it kept the pole and nothing else.

</details>

### 3.3 The centre of the pole

The same step as in 2.4, now on the real mask. The notebook gives you `rows, cols = np.nonzero(pole_mask)`. This step summarises the positions of 1289 pixels with two numbers.

<details markdown="1">
<summary>Show the implementation</summary>

```python
pole_row = rows.mean()
pole_col = cols.mean()
```

The centre of the pole lies at row 230.17 and column 305.15. The numbers carry decimals because they are averages of 1289 pixels, so the camera locates the pole more finely than a single pixel.

</details>

### 3.4 A function that finds any colour

Nothing to write here. The function `locate(image, colour)` packs 3.2 and 3.3 together, so that it works for any colour: it builds the mask of that colour and returns the average row and the average column. The notebook uses it to find the pole and the pivot. Keep the name `locate`, because the camera calls it on every photo.

### 3.5 The pole angle

You now have two points, the pivot and the centre of the pole, and the line between them is the pole. The notebook draws the figure below on a strongly tilted practice pole.

![Zoom on a tilted pole. A dashed vertical line rises from the pivot. The horizontal distance dx and the vertical distance dy go from the pivot to the pole centre, and the angle theta sits between the vertical and the pole]({{ '/practicals/inverted_pendulum_camera_vision/assets/angle_measurement.png' | relative_url }})

The angle θ is measured from the vertical, and the two distances form a right triangle.

- `dx` is how far the pole centre sits to the right of the pivot.
- `dy` is how far the pole centre sits above the pivot. Rows grow **downwards**, so the pole centre has the *smaller* row, and the subtraction goes the other way round.
- The tangent of θ is `dx / dy`, so the angle is its arctangent.

<details markdown="1">
<summary>Show the implementation</summary>

```python
dx = pole_col - pivot_col
dy = pivot_row - pole_row
pole_angle = math.atan2(dx, dy)
```

- `dx` is positive when the pole leans right and negative when it leans left.
- `dy` is `pivot_row - pole_row`, and not the other way round, because the row number decreases as you go up the photo.
- `math.atan2(dx, dy)` is the arctangent of `dx / dy`, in radians. It is used instead of `math.atan(dx / dy)` because it still gives the right angle when the pole falls past horizontal and `dy` becomes negative.

The camera measures 0.0348 rad, and the simulator reports 0.0359 rad. The small difference comes from the pixels, because the photo cannot show a position finer than its squares.

</details>

### 3.6 The camera

A single photo shows where the pole is. However, it cannot show how fast the pole moves. The photos form a sequence in time, and the difference between two consecutive samples gives a rate of change. The camera receives the photo from now and the photo from the previous step, taken 0.02 s earlier:

```
velocity ≈ (value now − value before) / 0.02 s
```

The functions `read_image` and `camera` are written for you, except one line: the angular velocity. The cart velocity, one line above it, shows you how.

<details markdown="1">
<summary>Show the implementation</summary>

```python
pole_angular_velocity = (angle_now - angle_before) / TIME_STEP
```

- `angle_now` and `angle_before` come from `read_image`, applied to the two photos.
- `TIME_STEP` holds 0.02, the time between two photos.

The camera numbers come close to the truth without matching it. Two photos give the average velocity between them. In contrast, the simulator reports the velocity at the second instant. Moreover, dividing by 0.02 multiplies every small pixel error by 50, which is why velocities measured by a camera are noisy.

</details>

## Part 4 — Your real cart

From now on the notebook films your own cart, painted in the colours chosen by your two digits.

### 4.1 Switch on the real camera

Paste your best weights and bias from Practical 2 and run the cell. **This run fails, and that is expected.** Your camera still looks for a red pole and a blue axle, your cart has other colours, and therefore `locate` finds no pixel at all. Without any measurement, the neuron pushes the same way at every step. Watch the pole fall, and continue with 4.2.

> If your weights are all zero, the neuron never looks at the pole, even when the camera works. Use the weights that balanced the pole in Practical 2.

### 4.2 The challenge: adapt the camera to your cart

This is the challenge of the practical, so this page gives no implementation. You already have the tools, because you read pixels in 3.1.

Your goal is to find one pixel inside your pole and one pixel inside your axle, using the row and column axes of the photo. The notebook gives you two lines of the form `POLE_COLOUR = real_image[0, 0]`. You only replace each `0, 0` with the row and the column of your pixel. Each line then stores the colour of that pixel in the variable. If the colour printed is `[255 255 255]`, you landed on the background. If it is `[0 0 0]`, you landed on the cart. Finally, check that `locate` returns a centre close to your pole instead of `nan`. `nan` means *not a number*. It appears when no pixel of that colour exists in the photo, because the average of an empty list is undefined. You have finished when the red squares sit on your pole and on your axle in the animation of 4.3. The notebook holds one hint if you need it.

### 4.3 Does your neuron see?

The cell runs your camera and your neuron together, once with the animation and then from five starting positions. If the average falls below 500, recalibrate one weight at a time, as in Practical 2, then register your result in the class form.

### 4.4 If you have time: is your camera telling the truth?

`compare_sensors` plots the four measurements of the simulator next to the four measurements of your camera during the same run. Comparing two sources of the same quantity is a standard check of data quality. Look for the measurement the camera gets almost perfectly, and the one that is the noisiest. The explanation at the end of 3.6 tells you why.

## Beyond the practical — The advanced levels

If you finish before the end of the session, the notebook continues with three levels. Real data rarely arrives clean, and each level degrades the photos in a new way. Each level also gives you less help than the one before. There are no implementations on this page, only the problem each level poses.

![The same cart under the four conditions: normal light, dim light, a deep shadow over part of the rail, and a noisy camera]({{ '/practicals/inverted_pendulum_camera_vision/assets/advanced_levels.png' | relative_url }})

| Level | What changes | Help in the notebook |
|---|---|---|
| 1 — The lights go dim | The brightness drops by a different amount on every run | A short introduction to the HSV colour description, guided steps and three hints |
| 2 — A shadow over the rail | Part of the rail lies in deep shadow | One hint |
| 3 — A cheap, noisy camera | Random noise on every pixel of every photo | A question, and no hints |

The notebook checks each level for you with `check_level(...)`. Register the highest level you pass in the class form.


## Quick reference

| Python | Meaning |
|---|---|
| `np.array([1, 2, 3])` | A vector: numbers in a row that support arithmetic all at once |
| `a * b` | Multiply two arrays element by element |
| `np.sum(a)` | Add all the numbers of an array |
| `a > 0` | Ask every number a question, which returns `True` or `False` for each |
| `grid.shape` | The size of a matrix or a photo: (rows, columns, layers) |
| `grid[r, c]` | The value in row `r`, column `c`, counting from zero |
| `image[:, :, 0]` | All rows, all columns, layer 0: the red layer of a photo |
| `(a == 1) & (b == 2)` | True only where both conditions are true |
| `rows, cols = np.nonzero(mask)` | The rows and the columns of the non-zero pixels |
| `rows.mean()` | The average of a list |
| `math.atan2(dx, dy)` | The angle whose tangent is `dx / dy`, in radians |
| `nan` | Not a number: the result of averaging an empty list |
