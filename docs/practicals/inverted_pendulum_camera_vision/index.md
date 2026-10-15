---
layout: default
title: "Practical 3 — Data manipulation with images"
---

# Practical 3 — Data manipulation with images

## What you will do today

In Practical 2, your neuron balanced the pole with four measurements that the simulator handed over, already clean and in the right order. Real systems rarely receive their data in such a convenient form. Instead, a sensor delivers raw data, and somebody has to turn it into the few numbers that a model can use.

Today the sensor is the simulation render, i.e., the image that the simulator draws 50 times per second. In this practical, you will refer to each of these images as a snapshot. For the computer, an image is a large table of numbers that you can also display on the screen. Consequently, every time you select, filter or average those numbers, you can look at the result and check it with your own eyes.

You will write the Python code that turns each snapshot into the four measurements of Practical 2. Then, the neuron will balance the pole using only the numbers that your code extracts from the snapshots. Finally, you will adapt that code to a cart of your own, painted in different colours.

**Two setups: a test setup and a validation setup.** The practical uses two versions of the same simulation. First, you build your code on a **test setup**, i.e., a practice cart with a red pole and a blue axle. In the test setup, the simulator also gives you the true measurements, so you can check every number that your code extracts. Then, you solve the challenge on a **validation setup**, i.e., a cart painted in colours that depend on your student number. In the validation setup, you adapt your code without a ready-made solution on this page. 

| Setup | Where in the practical | Cart | What you do |
|---|---|---|---|
| Test setup | Parts 0 to 3 | Red pole and blue axle, with the true measurements available | Build your code and check it against the truth |
| Validation setup | Part 4 and the advanced levels | Colours chosen by your student number | Adapt your code and balance the pole with it |

![A cart balancing a pole, controlled by a neuron that only sees snapshots of the simulation render. Red squares mark the pivot and the pole centre found in each snapshot]({{ '/practicals/inverted_pendulum_camera_vision/assets/camera_balanced.gif' | relative_url }})

The red squares in the animation mark the two points that your code finds in each snapshot, i.e., the pivot and the centre of the pole. Everything the neuron knows about the pole comes from those two points.

> **How to use this page.** Work in the notebook, where every cell marked `# YOUR CODE HERE` is yours to fill. The notebook tells you what to write in each of those cells. **Try it yourself first.** If you are stuck, come back to the matching section of this page and click **Show the implementation** to reveal the code and a line-by-line explanation. The challenge in 4.2 and the advanced levels have no implementation on this page, because those problems are yours to solve.

> **Before you begin.** Open the notebook, save your own copy to Drive, and rename it to `MEEN41490_P3_YourName`. If you need a reminder of these steps, see [Getting Started with Google Colab]({{ '/guides/getting-started-colab/' | relative_url }}).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RUWvuuhKjnhqFDpl_Bvi_g8Wbj_w9j8J)

## Part 0 — How a computer represents an image

Read this part before you open the notebook. It takes about five minutes, and every cell of the notebook relies on the ideas it introduces.

**A snapshot is an image.** Each snapshot of the simulation render is a digital image, like any image that you see on a computer screen. Therefore, everything in this part applies to any digital image, and not only to the snapshots of the simulator.

**An image is a grid of pixels.** A pixel is the smallest square of an image, and each pixel has a single colour. The snapshots in this practical have 400 rows and 600 columns of pixels, that is, 240 000 small squares in total. If you zoom far enough into any image, the individual squares become visible. For the computer, each square is a few numbers in a table, and the image on the screen is only one way of displaying that table.

![A snapshot of the test setup, and a zoom on a 12 by 12 square of it where the individual pixels are visible]({{ '/practicals/inverted_pendulum_camera_vision/assets/pixels_zoom.png' | relative_url }})

**A position is a row and a column.** Rows count from the top of the image **downwards**, and columns count from the left to the right. Both counts start at zero. Python always takes the row first, so `image[230, 305]` is the pixel in row 230 and column 305.

**A colour is three numbers.** Each pixel stores how much red, green and blue light it contains, on a scale from 0 (none) to 255 (full). Mixing these three amounts in different proportions gives every colour that the screen can show.

| Pixel | Colour |
|---|---|
| `[255, 0, 0]` | red |
| `[0, 0, 255]` | blue |
| `[255, 255, 255]` | white |
| `[0, 0, 0]` | black |

Consequently, a colour image is three grids of numbers stacked together, one grid per colour. For the snapshots of this practical, you write this size as `(400, 600, 3)`. You will refer to each of these grids as a layer. The figure below draws each layer of a snapshot of the test setup in grey, with white where that colour is at full strength and black where it is absent. The white background is bright in all three layers. In contrast, the red pole is bright only in the red layer, where it blends into the background, and it appears black in the other two layers.

![The red, green and blue layers of a snapshot of the test setup, each drawn in grey]({{ '/practicals/inverted_pendulum_camera_vision/assets/rgb_layers.png' | relative_url }})

**Finding an object means comparing every pixel with a reference colour.** The code that you write today to read the simulation render works in two steps. First, it compares the colour of every pixel in the snapshot with the colour of the pole. The code stores the results of these comparisons as a new image of `True` and `False` values, called a **mask**. Second, it averages the rows and the columns of the `True` pixels to obtain the **centre** of the pole, in the same way as you would compute a centre of mass.

![The mask of the pole: only the pixels with the colour of the pole are black]({{ '/practicals/inverted_pendulum_camera_vision/assets/pole_mask.png' | relative_url }})

With the centre of the pole and the position of the pivot, one line of trigonometry gives the pole angle. Then, two consecutive snapshots give the velocities, and these four numbers complete the measurement.

None of these operations is specific to images. Selecting values, comparing every value with a reference, counting the results and averaging the values that pass are the same steps you would apply to a log of temperature readings. The image only makes each of these steps visible on the screen.

This is all the background you need for today. Open the notebook with the **Open in Colab** button at the top of this page, and work through it from the setup cell onwards. Each section of this page carries the same number as one section of the notebook. Whenever you are stuck, come back to the matching section here and open **Show the implementation**.

## Part 1 — Setting up the notebook

Type the **last two digits of your student number** in the box of the setup cell, for example `37`, and run the cell. The cell prints `Ready.` when it finishes. Your two digits choose the colours of the cart in your validation setup, which you will meet in Part 4. Therefore, your validation setup will probably look different from the validation setup of your neighbour.

## Part 2 — Image representation

The four cells of this part introduce the operations that your code needs to read the simulation render. 

### 2.1 Vectors

A vector is a row of numbers, and the command `np.array(...)` stores one in a variable. A vector looks like the lists of Practical 2. However, you can multiply, add or compare a whole vector in one step. For example, `2 * np.array([1, 2, 3])` gives `[2 4 6]`. The cell asks you to compute the neuron of Practical 2 in this way.

<details markdown="1">
<summary>Show the implementation</summary>

```python
measurements = np.array([0.10, -0.30, 0.05, 0.40])
total = np.sum(weights * measurements) + bias
print("Total:", round(total, 2))
```

- `np.array([...])` turns the list of four measurements into an array.
- `weights * measurements` multiplies the two arrays element by element, that is, the first weight by the first measurement, the second weight by the second measurement, and so on.
- `np.sum(...)` adds the four products, and `+ bias` adds the bias. Together, these two steps write the neuron of Practical 2 in a single line.
- `round(total, 2)` keeps two decimals of the result. Without it, Python may print `0.31000000000000005`, which is in practice the same number. This happens because computers store decimals with a tiny rounding error.

</details>

### 2.2 Vector comparison

The expression `measurements > 0` compares every number with zero at once, and returns `True` or `False` for each one. Python counts `True` as 1 and `False` as 0. Therefore, the sum of the results is the number of `True` values. You will refer to this type of comparison as a filter. A filter is the most common operation on any dataset. Later, your code uses a filter to find the pole in each snapshot.

<details markdown="1">
<summary>Show the implementation</summary>

```python
print(measurements > 0)
print(np.sum(measurements > 0))
```

The first line prints `[ True False  True  True]`, which is one result per measurement. The second line adds the results, 1 + 0 + 1 + 1, and prints `3`.

</details>

### 2.3 Matrices

A matrix is a grid of numbers organised in rows and columns. Two kinds of brackets appear in this cell, and each kind has a different meaning.

| Brackets | Meaning | Example |
|---|---|---|
| square `[ ]` | read a position inside a matrix or vector | `grid[0, 1]` is the number in row 0, column 1 |
| round `( )` | run a function | `print(grid)`, `np.sum(grid)` |

The expression `grid.shape` takes no brackets, because it is not a function. Instead, it is a piece of information that the matrix carries with it, namely its size.

<details markdown="1">
<summary>Show the implementation</summary>

```python
print(grid.shape)
print(grid[1, 2])
```

- `grid.shape` is `(3, 5)`, which means 3 rows and 5 columns.
- `grid[1, 2]` reads the number in row 1 and column 2. Both counts start from zero, so this position is the second row and the third column. That position holds a `1`.

</details>

### 2.4 Matrix as an image

If you paint every 0 white and every 1 black, the matrix becomes an image in which each number is a pixel. In the matrix `canvas`, the 1s draw a small vertical pole.

![A 7 by 7 matrix drawn as an image: the 1s form a vertical pole in column 3]({{ '/practicals/inverted_pendulum_camera_vision/assets/canvas.png' | relative_url }})

The line `rows, cols = np.nonzero(canvas)` is already in the notebook. The function `np.nonzero` finds every pixel that is not zero and returns **two** lists, i.e., the rows of those pixels and their columns. Writing two names before the `=` stores the first list in `rows` and the second list in `cols`. In this example, `rows` is `[1 2 3 4 5]` and `cols` is `[3 3 3 3 3]`. Your task is to compute the average of each list. The canvas is small enough to check by eye, so you can confirm the method before you trust it on a snapshot of the test setup.

<details markdown="1">
<summary>Show the implementation</summary>

```python
print("Centre:", rows.mean(), cols.mean())
```

The command `.mean()` computes the average of a list. The average row is (1 + 2 + 3 + 4 + 5) / 5 = 3, and every column is 3. Therefore, the centre of the pole is in row 3 and column 3. This position matches the middle of the bar in the image.

</details>

## Part 3 — Building the code on the test setup

In this part, you build your code on the test setup. The test setup is a practice cart painted in simple colours, with a red pole and a blue axle. The axle is the pivot, that is, the hinge between the pole and the cart. The function `take_test_snapshot()` returns one snapshot together with the four true measurements at that instant. Consequently, you can check every result of your code against the truth before you move to the validation setup. Comparing a result with a known answer is the simplest way to validate any data processing.

### 3.1 Reading the size and the pixels of a snapshot

The cell asks you to print the size of the snapshot, draw the snapshot, and read two of its pixels.

<details markdown="1">
<summary>Show the implementation</summary>

```python
print(image.shape)
show_matrix(image)
print(image[0, 0])        # [255 255 255] -> white
print(image[230, 305])    # [255   0   0] -> red
```

- `image.shape` is `(400, 600, 3)`, which means 400 rows, 600 columns and 3 colour layers.
- `show_matrix(image)` draws the snapshot with its row and column axes. You will use those axes again in Part 4.
- `image[0, 0]` is the top-left pixel, which belongs to the background. It holds `[255 255 255]`, which is **white**.
- `image[230, 305]` falls inside the pole. It holds `[255 0 0]`, which is **red**.

![A snapshot of the test setup with its row and column axes]({{ '/practicals/inverted_pendulum_camera_vision/assets/practice_photo.png' | relative_url }})

</details>

### 3.2 Finding every pole pixel with a mask

The line `red = image[:, :, 0]` picks the whole red layer of the snapshot. The colon `:` means *all of them*, so the expression reads *all rows, all columns, layer 0*. The notebook does the same for the green and the blue layers.

A pixel belongs to the pole only when its red **and** its green **and** its blue values match the pole colour. The notebook gives you a line in which the green and the blue conditions are already joined by `&`, the Python symbol for *and*. Run that line as it is first. The resulting mask also keeps the black cart and the rail, because black has no green and no blue either. That is, the red condition is still missing.

<details markdown="1">
<summary>Show the implementation</summary>

```python
pole_mask = (green == POLE_COLOUR[1]) & (blue == POLE_COLOUR[2]) & (red == POLE_COLOUR[0])
```

- `green == POLE_COLOUR[1]` compares the green value of every pixel with 0. The result is a grid of `True` and `False` values. The blue condition makes the same comparison on the blue layer.
- `red == POLE_COLOUR[0]` compares the red value of every pixel with 255. The black cart fails this comparison, so the new condition removes the cart from the mask.
- `&` keeps a pixel `True` only when all three comparisons are `True`. Each comparison needs its own round brackets. Otherwise, Python mixes up the order of the operations.

With the red condition, the mask drops from 3276 to 1289 `True` pixels, which is the area of the pole in pixels. The image of the mask shows the result of the filter, so you can see at once whether it kept the pole and nothing else.

</details>

### 3.3 Locating the centre of the pole

This cell repeats the step of 2.4, now on the real mask. The notebook gives you the line `rows, cols = np.nonzero(pole_mask)`. Your code then summarises the positions of 1289 pixels with only two numbers.

<details markdown="1">
<summary>Show the implementation</summary>

```python
pole_row = rows.mean()
pole_col = cols.mean()
```

The centre of the pole lies at row 230.17 and column 305.15. These numbers carry decimals because they are averages over 1289 pixels. Consequently, your code locates the pole more finely than the size of a single pixel.

</details>

### 3.4 A function that locates any colour

There is nothing to write in this cell. The function `locate(image, colour)` packs the steps of 3.2 and 3.3 together, so that they work for any colour. For a given colour, the function builds the mask and returns the average row and the average column of that mask. The notebook uses `locate` to find both the pole and the pivot. Keep the name `locate` unchanged, because the function `measure` calls it on every snapshot.

### 3.5 Measuring the pole angle

You now have two points, i.e., the pivot and the centre of the pole, and the line between them represents the pole. The notebook draws the figure below on a strongly tilted pole of the test setup.

![Zoom on a tilted pole. A dashed vertical line rises from the pivot. The horizontal distance dx and the vertical distance dy go from the pivot to the pole centre, and the angle theta sits between the vertical and the pole]({{ '/practicals/inverted_pendulum_camera_vision/assets/angle_measurement.png' | relative_url }})

You measure the angle θ from the vertical, and the two distances in the figure form a right triangle.

- `dx` is how far the pole centre sits to the right of the pivot.
- `dy` is how far the pole centre sits above the pivot. Rows grow **downwards**, so the pole centre has the *smaller* row number. For this reason, the subtraction for `dy` goes the other way round.
- The tangent of θ is `dx / dy`, so the angle is the arctangent of that ratio.

<details markdown="1">
<summary>Show the implementation</summary>

```python
dx = pole_col - pivot_col
dy = pivot_row - pole_row
pole_angle = math.atan2(dx, dy)
```

- `dx` is positive when the pole leans right and negative when it leans left.
- `dy` is `pivot_row - pole_row`, and not the other way round, because the row number decreases as you go up the snapshot.
- `math.atan2(dx, dy)` is the arctangent of `dx / dy`, in radians.

Your code measures 0.0348 rad, whereas the simulator reports 0.0359 rad. The small difference comes from the pixels, because the snapshot cannot show a position finer than its squares.

</details>

### 3.6 Estimating velocities from two consecutive snapshots

A single snapshot shows where the pole is. However, it cannot show how fast the pole moves. The snapshots form a sequence in time, and the difference between two consecutive samples gives a rate of change. For this reason, the function `measure` receives the current snapshot and the snapshot from the previous step, taken 0.02 s earlier. The velocity then follows from this equation:

```
velocity ≈ (value at the current step − value at the previous step) / 0.02 s
```

The functions `read_image` and `measure` are already written for you, except for one line that computes the angular velocity. The line just above it computes the cart velocity and shows you the pattern to follow.

<details markdown="1">
<summary>Show the implementation</summary>

```python
pole_angular_velocity = (angle_current - angle_previous) / TIME_STEP
```

- `angle_current` and `angle_previous` come from `read_image`, applied to the two snapshots.
- `TIME_STEP` holds 0.02, which is the time in seconds between two snapshots.

The velocities that your code estimates come close to the truth without matching it exactly. Two snapshots give the average velocity over the interval between them. In contrast, the simulator reports the velocity at the second instant. Moreover, dividing by 0.02 multiplies every small pixel error by 50. This amplification is the reason why velocities measured from a simulation render are noisy.

</details>

## Part 4 — The challenge on the validation setup

From now on, the notebook switches to your validation setup. It renders your own cart, painted in the colours that your two digits selected in Part 1. In contrast with the test setup, this page gives you no solution for the colours of your cart.

### 4.1 Running the code of the test setup on the validation setup

Paste your best weights and bias from Practical 2 into the cell and run it. **This run fails, and that failure is expected.** Your code still looks for the red pole and the blue axle of the test setup, whereas your validation setup has other colours. Therefore, `locate` finds no pixel at all in the snapshot. Without any measurement, the neuron pushes the cart in the same direction at every step. Watch the pole fall, and then continue with 4.2.

> If your weights are all zero, the neuron never looks at the pole, even when your code reads the render correctly. Use the weights that balanced the pole in Practical 2.

### 4.2 Challenge — adapt the code to the colours of your validation setup

This is the challenge of the practical, so this page gives no implementation for it. You already have the necessary tools, because you read individual pixels in 3.1.

Your goal is to find one pixel inside your pole and one pixel inside your axle, using the row and column axes of the snapshot. The notebook gives you two lines of the form `POLE_COLOUR = validation_image[0, 0]`. In each line, you only replace `0, 0` with the row and the column of your chosen pixel. Each line then stores the colour of that pixel in the variable.

Check the colour that the notebook prints for each pixel. If it is `[255 255 255]`, you landed on the background. If it is `[0 0 0]`, you landed on the cart. Then, check that `locate` returns a centre close to your pole, and not `nan`. The value `nan` means *not a number*. It appears when the snapshot contains no pixel of the requested colour. In that case, the average of an empty list is undefined.

You have finished the challenge when the red squares sit on your pole and on your axle in the animation of 4.3. The notebook holds one hint if you need it.

### 4.3 Testing the neuron with your code

The cell runs your code and your neuron together, first once with the animation and then from five different starting positions. If the average score falls below 500, recalibrate one weight at a time, as you did in Practical 2. Finally, register your result in the class form.

### 4.4 Optional — comparing your measurements with the simulator

The function `compare_sensors` plots the four measurements of the simulator next to the four measurements of your code during the same run. Comparing two sources of the same quantity is a standard check of data quality. In the plot, look for the measurement that your code gets almost perfectly and for the measurement that is the noisiest. The explanation at the end of 3.6 tells you why this difference appears.

## You have finished the practical

**Well done! You have finished every required task of Practical 3.** Section 4.4 and the advanced levels below are optional. If you want to continue, follow the notebook into the advanced levels.

## Beyond the practical — Optional advanced levels with degraded snapshots

If you finish before the end of the session, the notebook continues with two optional advanced levels. Real data rarely arrives clean, and each level degrades the snapshots of your validation setup in a new way. Additionally, the second level gives you less help than the first one. This page contains no implementations for these levels, only a description of the problem that each level poses.

**Introduction to the advanced levels.** Your `locate` function looks for exact RGB colours. However, when the light changes, every number of every pixel changes with it. Therefore, the notebook first introduces the Hue, Saturation and Value (HSV) colour description. HSV separates *which* colour a pixel has from *how bright* the pixel is. A short exploration cell lets you compare the same pole pixel in normal and in dim light, in RGB and in HSV.

![The same cart under three conditions: normal light, a deep shadow over part of the rail, and a noisy render]({{ '/practicals/inverted_pendulum_camera_vision/assets/advanced_levels.png' | relative_url }})

| Level | What changes | Help in the notebook |
|---|---|---|
| 1 — A shadow over the rail | The light is dimmer, and part of the rail lies in deep shadow | The HSV introduction and four hints |
| 2 — A noisy simulation render | Random noise on every pixel of every snapshot | A question, and no hints |

The notebook checks each level for you with the function `check_level(...)`. Register the highest level that you pass in the class form.


## Quick reference of Python commands

| Python | Meaning |
|---|---|
| `np.array([1, 2, 3])` | A vector, i.e., numbers in a row that support arithmetic all at once |
| `a * b` | Multiply two arrays element by element |
| `np.sum(a)` | Add all the numbers of an array |
| `a > 0` | Compare every number with 0, which returns `True` or `False` for each |
| `grid.shape` | The size of a matrix or a snapshot, as (rows, columns, layers) |
| `grid[r, c]` | The value in row `r` and column `c`, counting from zero |
| `image[:, :, 0]` | All rows, all columns and layer 0, i.e., the red layer of a snapshot |
| `(a == 1) & (b == 2)` | True only where both conditions are true |
| `rows, cols = np.nonzero(mask)` | The rows and the columns of the non-zero pixels |
| `rows.mean()` | The average of a list |
| `math.atan2(dx, dy)` | The angle whose tangent is `dx / dy`, in radians |
| `nan` | Not a number, i.e., the result of averaging an empty list |
