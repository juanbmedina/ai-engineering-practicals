---
layout: default
title: "Practical 2 — IA is not magic"
---

# Practical 2 — IA is not magic

<!-- <p class="subtitle"><strong>MEEN41490 — AI in Engineering</strong></p> -->

## What you will do today

Every AI system you have heard about is built from one small component repeated many times. The model that writes text, the model that reads a radiograph, and the model that sorts parts on a production line all share it. That component is an artificial neuron, and it performs a multiplication, an addition, and a comparison. Today you will use exactly one of them.

You will optimize a single neuron to keep a pole balanced on top of a moving cart. You will write it in Python inside Google Colab, a web platform that runs your code on a machine in a Google data centre and returns the result to your browser.

![A cart moving along a rail, keeping a pole upright, controlled by a single neuron](/practicals/inverted_pendulum_neuron_control/assets/cartpole_balanced.gif)

The pole in that animation stays up because a trained model controls the system. A trained model is a mathematical expression plus a set of numbers found by an optimisation process. Finding those numbers is the goal of many machine learning techniques. Today you will find the numbers by hand, acting as an optimizer.


> **How to use this page.** The page contains the explanations and some parts of the code. The notebook contains the complementary code structures needed to make the simulation work. In some parts of the notebook the code has been removed and marked `# YOUR CODE HERE`. Read the explanation here, then write the code there.

> **Before you begin.** If this is your first time in Google Colab, read [Getting Started with Google Colab](/guides/getting-started-colab/) first. It covers opening the shared notebook, saving your own copy to Drive, and finding your way around the Colab interface. Come back here once your copy is open and the setup cell has run.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19KKoM1SPAnatIGxEffiK24M9scgGGzDd)

## Part 0 — The problem

A cart runs along a straight rail, with a rigid pole attached to its top through a free-turning hinge. The pole behaves as an inverted pendulum, that is, a pendulum standing above its pivot instead of hanging below it.

An ordinary pendulum returns to the bottom on its own, because gravity pulls it back towards the lowest point. An inverted pendulum does the opposite. Vertical is still an equilibrium, however it is an unstable one, and the smallest tilt grows into a larger tilt. Gravity now pulls the pole further away from vertical instead of back towards it.

Nothing can hold the pole from above, and the hinge applies no torque. Consequently, the only way to recover a falling pole is to moving the cart. You already know the manoeuvre, because it is what your hand does when you balance a broom on your palm. The broom leans forward and your hand moves forward.

![Schematic of the cart and pole, showing cart position, cart velocity, pole angle and angular velocity, and the two possible pushes](/practicals/inverted_pendulum_neuron_control/assets/cartpole_system.svg)

At every instant the simulation reports four measurements of the system.

| Measurement | Meaning | Sign |
|---|---|---|
| Cart position | Distance of the cart from the centre of the rail | Positive to the right |
| Cart velocity | Speed of the cart along the rail | Positive moving right |
| Pole angle | Tilt of the pole away from vertical | Positive leaning right |
| Pole angular velocity | Rate at which the tilt changes | Positive falling right |

You control the cart with one command. That command is the direction of a push with a fixed force of 10 N, and doing nothing is not available. The simulation asks you for a direction every 0.02 s, so you make 50 decisions every second.

The run ends under three conditions. The pole passes 12 degrees from vertical, or the cart reaches the end of the rail at 2.4 m from the centre, or 500 steps elapse. Therefore, 500 steps is a perfect score, and the number of steps survived is your only measure of quality.

## Part 1 — The rule that decides the push

Engineers have solved this problem for a long time, and several methods work. Classical control theory offers the PID controller, which reacts to the error, to its accumulated history, and to its rate of change. It also offers the linear quadratic regulator (LQR), which derives the best possible reaction from a model of the physics. Reinforcement learning takes a different route, because it discards the model of the physics and improves a controller from repeated attempts.

Those methods disagree about how to obtain the controller. However, they agree about its shape. Each one produces a single number from the measurements, and the sign of that number selects the action.

That is the shape you will build. You take the four measurements, you multiply each of them by a number of your choice, and you add the four products together. You add one final number, called the bias, which shifts the result up or down. The sum is a single value, and you compare it with zero.

```
u = w1 * cart_position
  + w2 * cart_velocity
  + w3 * pole_angle
  + w4 * pole_angular_velocity
  + b

if u > 0, push right.  Otherwise, push left.
```

The four numbers `w1` to `w4` are called weights, and each one states how much importance the rule gives to one measurement. A weight of zero removes that measurement from the decision entirely. A large weight makes the decision follow that measurement almost alone. A negative weight reverses the meaning of the measurement.

Before you write any code, make one prediction. Suppose the pole angle is the only measurement different from zero, and the pole leans to the right. In that situation, the cart must move in one direction to push the pole back upright. Decide which direction that is, then decide whether the weight on the pole angle must be positive or negative to produce that direction.

Note this: rule contains no equations of motion, no mass, no length of the pole, and no gravity. It knows nothing about pendulums. Four multiplications and one comparison decide every push, and the entire behaviour of the system hides in the five numbers.

## Part 2 — Introduction to Python

Each subsection below matches one cell in the notebook. Type the code instead of pasting it, because typing forces you to read every character.

### 3.1 A variable

A variable is a name attached to a value that the machine holds in memory. You create one with the `=` sign, and the name goes on the left. The `print` function displays a value on the screen.

```python
pole_angle = 0.05
print(pole_angle)
```

Run the cell with **Shift + Enter**. The number 0.05 is the tilt of the pole in radians, which is about 3 degrees to the right.

Names in Python may contain letters, numbers and underscores, and they carry no meaning for the machine. The name `pole_angle` is exactly as valid as the name `q7`, and the first name is better because the next reader understands it immediately.

### 3.2 Arithmetic

A variable holds a value, so it can take part in a calculation. The result of that calculation is a value like any other, and you can store it under a new name.

```python
weight_pole_angle = 1.0
contribution = pole_angle * weight_pole_angle
print(contribution)
```

The symbols `+`, `-`, `*` and `/` perform the four arithmetic operations, and `*` is the multiplication sign. You have now computed one term of the rule from Part 1.

This cell runs correctly only because the previous cell ran before it. The name `pole_angle` exists in memory because you created it there.

### 3.3 The whole sum

The rule needs four measurements, four weights, and the bias. The following cell contains one instant of the simulation — the cart sits slightly right of centre and moves left, and the pole leans right and falls right.

```python
cart_position = 0.10
cart_velocity = -0.30
pole_angle = 0.05
pole_angular_velocity = 0.40

weight_cart_position = 0.1
weight_cart_velocity = 0.5
weight_pole_angle = 1.0
weight_pole_angular_velocity = 1.0
bias = 0.0

total = (cart_position * weight_cart_position
         + cart_velocity * weight_cart_velocity
         + pole_angle * weight_pole_angle
         + pole_angular_velocity * weight_pole_angular_velocity
         + bias)

print("Total:", total)
```

The brackets around the sum let one instruction span several lines, and they keep the code readable. The `print` function accepts several values separated by commas, and it displays them on one line.

The cell prints `0.31`. The value is positive, and the rule therefore pushes right. The pole leans right and falls right, so pushing right moves the cart under the pole. The rule agrees with the physics in this instant.

### 3.4 The decision

A comparison such as `total > 0` produces an answer of true or false, and `if` runs a block of code only when the answer is true. The `else` block runs in the opposite case.

```python
if total > 0:
    action = 1
else:
    action = 0

print("Action:", action)
```

The four spaces at the start of the indented lines are part of the language. Python uses indentation to mark which lines belong to the `if` and which lines belong to the `else`, and a missing indent is an error.

The simulator expects `1` for a push to the right and `0` for a push to the left. Your rule now converts four measurements into one legal command.

### 3.5 Lists

Writing eight separate names works once, however the simulator hands you the four measurements together, and you need them under a single name. A list is an ordered collection of values stored under one name, and square brackets define it.

```python
measurements = [0.10, -0.30, 0.05, 0.40]
weights = [0.1, 0.5, 1.0, 1.0]

print(measurements[2])
print(weights[2])
```

Square brackets after the name select one element. Python counts from zero, so `measurements[0]` is the cart position and `measurements[3]` is the angular velocity. The cell above prints the pole angle and the weight that multiplies it.

The order of the list is a convention that you must respect. Position, velocity, angle, angular velocity — the simulator always uses that order, and your weights must follow it.

### 3.6 A function

You need the same calculation 50 times per second, and copying it 50 times is not an option. A function is a named block of code that accepts values, performs its work, and returns a result. You define it once with `def`, and you use it as often as you need.

```python
def controller(measurements, weights, bias):
    total = (measurements[0] * weights[0]
             + measurements[1] * weights[1]
             + measurements[2] * weights[2]
             + measurements[3] * weights[3]
             + bias)
    if total > 0:
        return 1
    else:
        return 0


print(controller([0.10, -0.30, 0.05, 0.40], [0.1, 0.5, 1.0, 1.0], 0.0))
```

The names `measurements`, `weights` and `bias` are the arguments of the function — that is, the values it expects each time it runs. They exist only inside the function, and they take whatever values you supply in the call. The `return` statement ends the function and hands one value back.

The last line calls the function with the same numbers as before, and it prints `1`. Your function reproduces the result you computed by hand in the previous cells.

The simulator will call this function on every step of the run, so the name matters. Keep it as `controller`, because the helper functions from the setup cell look for exactly that name.

## Part 3 — The rule as a neuron

The rule you wrote is an artificial neuron, and the diagram below is the standard way of drawing it.

![Diagram of a single artificial neuron: four inputs, four weights, a bias, a summation, and a step function producing the action](/practicals/inverted_pendulum_neuron_control/assets/neuron.svg)

Every part of the diagram corresponds to something you typed.

| In the diagram | In your code | What it does |
|---|---|---|
| Inputs | `measurements` | The four numbers the neuron receives |
| Weights | `weights` | How much each input counts |
| Bias | `bias` | A constant offset added to the sum |
| Summation | `total = ...` | Adds the weighted inputs |
| Activation function | `if total > 0` | Turns the sum into an output |

The comparison with zero is an activation function — that is, the step applied to the sum before the neuron reports its output. This neuron uses the simplest one, which returns one of two values. 

Three properties separate your neuron from a modern model, and none of them changes the arithmetic.

**Count.** You have one neuron with four inputs. A model that recognises objects in an image has millions of neurons, and a large language model has hundreds of billions of weights. Each of those weights sits in a multiplication similar to yours.

**Arrangement.** Your neuron reports the action directly. In a larger model the output of one neuron becomes the input of the next, and the neurons form layers. That arrangement is what the word *deep* refers to in deep learning.

**Origin of the numbers.** You are about to choose four weights by hand. Nobody chooses billions of weights by hand, so the numbers come from a training procedure that adjusts them automatically.

## Part 4 — Calibrate your neuron

The setup cell defined three helpers, and all three call the `controller` function you wrote.

| Helper | What it does |
|---|---|
| `show(weights, bias)` | Plays one run, prints the number of steps, and displays the animation |
| `average_steps(weights, bias)` | Plays five runs from five different starting positions and returns the average |
| `run_episode(weights, bias)` | Plays one run and returns the number of steps, without the animation |

Start from a neuron that ignores everything.

```python
my_weights = [0.0, 0.0, 0.0, 0.0]
my_bias = 0.0

show(my_weights, my_bias)
```

The run lasts about ten steps. All four products are zero, the total is zero, and zero is not greater than zero, so the neuron pushes left at every step without ever looking at the pole.

![A cart-pole run in which the pole falls immediately](/practicals/inverted_pendulum_neuron_control/assets/cartpole_falling.gif)

Now calibrate. Change one weight, run the cell again, and read the number of steps. Change one number at a time, because changing several at once tells you nothing about which change helped.

Three suggestions guide the search.

1. Start with the pole angle, which is the third weight. Your answer to the prediction in Part 1 gives you its sign.
2. Then add the angular velocity, which is the fourth weight. The angle reports where the pole is, and the angular velocity reports where it is going.
3. Keep the bias at zero at first. The bias is a constant preference for one direction, and a large bias makes the neuron push the same way regardless of the measurements.

<!-- ### One run proves very little

A single run starts from one particular position, and that position may have been favourable. The next cell replays your five numbers from five different starting positions and reports the average.

```python
print("Average over five starts:", average_steps(my_weights, my_bias))
```

A setting that scores 500 once and 90 on average has not solved the problem. It was lucky. Report the average, not the best run, and the same rule applies to every model you will ever evaluate.

### Sweep one weight automatically

You have been changing a number and running a cell by hand. A loop does that work for you. A `for` loop takes a list and repeats the same block of code once for each element, and the variable after `for` holds the current element.

```python
values = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
scores = []

for value in values:
    weights = [0.0, 0.0, 1.0, value]
    score = average_steps(weights, 0.0)
    scores.append(score)
    print("angular velocity weight:", value, "  average steps:", score)
```

The list `scores` starts empty, and `scores.append(score)` adds one result to its end on every pass. The cell keeps the weight on the pole angle fixed at 1.0 and varies only the weight on the angular velocity.

```python
import matplotlib.pyplot as plt

plt.plot(values, scores, marker="o")
plt.xlabel("Weight on the angular velocity")
plt.ylabel("Average steps survived")
plt.title("Effect of one weight")
plt.grid(True)
plt.show()
```

The `import` statement loads a library — that is, a collection of code written by somebody else. Matplotlib draws charts, and you did not install it. Colab arrives with several hundred common libraries already present, which is the main practical reason this module uses it instead of asking you to configure Python on your own laptop.

The curve rises steeply and then falls, and the two ends fail for different reasons. At zero, the neuron reacts to the tilt with no information about how fast that tilt grows, and the pole passes 12 degrees before the cart arrives underneath it. At 4.0, the neuron holds the pole almost vertical, however it does so on a cart that drifts steadily in one direction, and the run ends when the cart reaches the end of the rail.

That second failure explains the two weights you left at zero. The cart position and the cart velocity enter the rule to keep the cart near the centre, and they earn small weights compared with the two terms that fight gravity.

The shape of the curve is also the reason calibration is hard. More of a good thing stops helping at some point, and the best setting sits at a place you cannot read off the physics. -->


Two of the four measurements carry almost all of the useful information. The pole angle reports which way the pole leans, and the angular velocity reports how fast the lean grows. Consequently, the rule that works reduces to a short sentence. Push in the direction the pole is falling. That is as far as the hints go.

> **The challenge:**  The exact numbers are not given here — finding them is the challenge. ***The first person in the room to reach an average of 500 steps over five starting positions wins.*** Keep calibrating.

<!-- ## Part 5 — Let the machine calibrate it

You spent several minutes adjusting numbers by hand. The machine does the same job by generating settings at random and keeping whichever one scores highest. Run the last cell of the notebook and watch the printed scores climb.

```python
generator = np.random.default_rng(0)
best_score = -1
best_weights = None
best_bias = 0.0

for attempt in range(1, 301):
    candidate_weights = [float(w) for w in generator.uniform(-2, 2, 4)]
    candidate_bias = float(generator.uniform(-0.1, 0.1))
    score = average_steps(candidate_weights, candidate_bias)
    if score > best_score:
        best_score = score
        best_weights = candidate_weights
        best_bias = candidate_bias
        print("attempt", attempt, "  average steps", round(score, 1))
    if best_score >= 500:
        break

print()
print("Best weights:", [round(w, 2) for w in best_weights])
print("Bias:", round(best_bias, 2))
show(best_weights, best_bias)
```

The search usually reaches 500 steps within a hundred attempts, and it takes a few seconds.

The machine used no understanding of pendulums. It generated four numbers at random, played five runs, kept the setting when the score improved, and repeated. That is the entire method, and it is the same thing you did with your hands and your eyes.

Two observations are worth making before you leave.

First, the numbers the machine found are almost certainly not the numbers you found, and both settings work. Many settings solve this problem, so there is no single correct answer to recover.

Second, random search succeeds here because there are only five numbers to choose. Doubling the count makes the search far harder, and a model with a million numbers cannot be calibrated this way at any speed. That limitation is the reason gradient descent exists, and gradient descent is the subject of the lecture *How a model learns* in Week 4. -->

<!-- ## Before you leave

- Your own copy of the notebook sits in your Google Drive with your name in the title
- You can explain what a variable, a list and a function are, in one sentence each
- Your `controller` function returns 0 or 1 from four measurements and five numbers
- You kept the pole up for 500 steps with weights you chose yourself
- You reported an average over five starting positions instead of your best single run
- You can state what the random search did, and what it did not do

## Optional, after the session

1. Set every weight except the pole angle to zero and find the best bias. How many steps can a neuron survive on one measurement?
2. Take your best setting and multiply all four weights by 10, keeping the bias at zero. Predict the new score before you run it, then explain the result.
3. Give the machine a harder job. Replace `average_steps(candidate_weights, candidate_bias)` with a version that uses ten starting positions, and check whether the winning setting changes.
4. Change the seed inside `show` to a number of your choice and check whether your setting still holds from that starting position. -->

## Quick reference

| Action | How |
|---|---|
| Run the cell and move down | Shift + Enter |
| Run the cell and stay | Ctrl / Cmd + Enter |
| New code cell below | Ctrl + M, then B |
| Delete the selected cell | Ctrl + M, then D |
| Undo, including a deleted cell | Ctrl + Z |
| Run every cell | Runtime → Run all |
| Clear the memory | Runtime → Restart session |
| Find the file in Drive | File → Locate in Drive |

| Python | Meaning |
|---|---|
| `name = 4.5` | Store a value under a name |
| `print(a, b)` | Display values on the screen |
| `a * b`, `a + b` | Multiply, add |
| `values = [1, 2, 3]` | A list of values in order |
| `values[0]` | The first element, counting from zero |
| `values.append(4)` | Add one element at the end |
| `if a > 0:` | Run the indented block when the comparison is true |
| `def f(x):` | Define a function that accepts `x` |
| `return y` | End the function and hand back `y` |
| `for v in values:` | Repeat the indented block once per element |
| `import matplotlib.pyplot as plt` | Load a library of code written by somebody else |

