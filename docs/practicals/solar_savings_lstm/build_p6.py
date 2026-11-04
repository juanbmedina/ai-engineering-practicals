"""Build the notebook for Practical 6 (Forecasting solar radiation with an LSTM to estimate the savings and payback of rooftop panels).

Run:  python3 build_p6.py            -> MEEN41490_P6_Solar_Savings_STUDENT.ipynb
      python3 build_p6.py --teacher  -> MEEN41490_P6_Solar_Savings_TEACHER.ipynb
"""
import json, sys

TEACHER = "--teacher" in sys.argv
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text, title=None, hidden=False):
    src = text.strip("\n")
    if title:
        src = f"# @title {title}\n" + src
    meta = {"cellView": "form"} if hidden else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None,
                  "outputs": [], "source": src})


# =========================================================================== intro
md(r"""
# Practical 6 — Forecasting solar radiation with an LSTM to estimate the savings and payback of rooftop panels

**AI in Engineering (MEEN41490)**

Four families in Dingle, Co. Kerry, want solar panels to pay less for electricity. Each family asks an engineer two questions. How much electricity would panels on our roof produce in a year? And when do we recover what we pay for them? You are the engineer of one of the families.

You answer with three things. The **solar radiation** at the house, hour by hour, obtained from satellites and weather models. A **formula** that turns the radiation on a roof into electricity from a panel. And the family's **electricity records**, i.e., how much they use and at what time of the day. In Parts 1 to 3, you build a model that forecasts the solar radiation. In Part 4, you turn the forecast into energy and into money. In Part 5, you compare your estimate with what the panels of the family really produced once installed.

**How this notebook works.** Run the cells in order. To run a cell, click the play button on its left. In Part 3 you edit code directly, so read Parts 1 and 2 with care. Part 0 of the practical page explains where the data come from, the curve of a solar panel, the curve of a house, the formula from radiation to electricity, and how the LSTM reads a window of hours.

### The data of the practice

The notebook uses two data files, which sit on the practical page under "The data of the session". `dingle_weather_2010_2021.csv` holds the hourly weather of Dingle from Open-Meteo. `house1.csv` holds the records of the family, every 30 minutes, from the study of Avgoloupis et al. (2026). Every team works with the same family, house 1 of the study. You do not need to download or upload anything. The setup cell downloads the two files into Colab and prepares the helper functions. If the course website is not reachable, the cell fetches the same data from their original sources, Open-Meteo and Figshare.
""")

# =========================================================================== setup
code(r"""
HOUSE = 1          # the family of this session (house 1 of the study), the same for every team

# ---- Libraries -------------------------------------------------------------
import os, json, time, urllib.request
import numpy as np                 # NumPy: arrays of numbers
import pandas as pd                # pandas: tables with a time index
import matplotlib.pyplot as plt    # Matplotlib: charts
import torch                       # PyTorch: the library that builds and trains the model
import torch.nn as nn
torch.set_num_threads(2)

# ---- Data files ------------------------------------------------------------
# The cell downloads the two data files from the course website into Colab. It skips any
# file that is already in the Files panel, e.g., when you run this cell a second time.
# If the course website does not answer, it builds the same files from the original sources:
# the Open-Meteo archive for the weather and Figshare for the records of the family.
DATA_URL = "https://juanbmedina.github.io/ai-engineering-practicals/practicals/solar_savings_lstm/assets/data/"
WEATHER_FILE = "dingle_weather_2010_2021.csv"
HOUSE_FILE = f"house{HOUSE}.csv"
FILES = [WEATHER_FILE, HOUSE_FILE]
WEATHER_COLUMNS = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance",
                   "global_tilted_irradiance", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
                   "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
FIGSHARE_IDS = {1: 61423177, 2: 61423180, 3: 61423183, 4: 61423189}     # the four household files of the study

def download_from_course_site(name):
    urllib.request.urlretrieve(DATA_URL + name, name + ".part")
    os.rename(name + ".part", name)                     # only complete downloads get the final name

def build_weather_from_open_meteo():
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=52.141&longitude=-10.269"
           "&start_date=2010-01-01&end_date=2021-12-31&hourly=" + ",".join(WEATHER_COLUMNS) +
           "&tilt=30&azimuth=0&timezone=UTC&wind_speed_unit=ms")
    with urllib.request.urlopen(url, timeout=120) as response:
        hourly = json.load(response)["hourly"]
    frame = pd.DataFrame({"time": hourly["time"], **{c: hourly[c] for c in WEATHER_COLUMNS}})
    frame.to_csv(WEATHER_FILE, index=False)

def build_house_from_figshare(number):
    url = f"https://ndownloader.figshare.com/files/{FIGSHARE_IDS[number]}"
    raw = pd.read_csv(url)
    frame = pd.DataFrame({"timestamp": raw["timestamp"].str.slice(0, 16), "solar_kWh": raw["solar_kWh"],
                          "import_kWh": raw["as230_import_kWh"], "export_kWh": raw["as230_export_kWh"],
                          "elec_price": raw["elec_price"]})
    frame.to_csv(f"house{number}.csv", index=False)

for name in FILES:
    if os.path.exists(name):
        continue
    try:
        download_from_course_site(name)
    except Exception as e:
        print(f"Course website: could not download {name} ({e}). Trying the original source...")
        try:
            if name == WEATHER_FILE:
                build_weather_from_open_meteo()
            else:
                build_house_from_figshare(HOUSE)
        except Exception as e2:
            print(f"Original source: could not build {name} ({e2}).")
missing = [n for n in FILES if not os.path.exists(n)]
if missing:
    raise SystemExit(f"Could not get {missing}. Run this cell again. If the error repeats, download "
                     "the files from the practical page, upload them in the Files panel (folder icon on the left) "
                     "and run this cell once more.")

# ---- The years of the exercise --------------------------------------------
TRAIN_YEARS = (2010, 2017)      # the model learns from these years
VALID_YEAR  = 2018              # the model is scored on this year during the challenge
REAL_YEAR   = 2019              # the year of the estimate, and the year of the real measurements
TARGET_MAE  = 18                # the target of the challenge, in W/m² (the MAE must be below it)

# ---- Helper functions ------------------------------------------------------
# These functions draw the figures and print the tables of the notebook. The cells of
# Parts 1 to 5 call them. Double-click the title of this cell if you want to read them.

def load_weather():
    frame = pd.read_csv("dingle_weather_2010_2021.csv", parse_dates=["time"]).set_index("time")
    frame = frame[frame.index.year <= REAL_YEAR]
    frame["hour_of_day"] = frame.index.hour            # 0 to 23
    frame["day_of_year"] = frame.index.dayofyear       # 1 to 366
    return frame

def load_house(number):
    frame = pd.read_csv(f"house{number}.csv", parse_dates=["timestamp"]).set_index("timestamp")
    frame = frame.drop(columns=["elec_price"])         # a price scenario of the study, not a real bill
    return frame

def plot_week(frame, start, columns, title, unit):
    # One week of the chosen columns, from the date in start.
    # Both weeks use the same axes, 7 days on x and 0 to 1000 W/m² on y, so that you can compare them.
    first = pd.Timestamp(start); last = first + pd.Timedelta(days=7)
    week = frame.loc[first:last]
    fig, ax = plt.subplots(figsize=(11, 3))
    for column in columns:
        ax.plot(week.index, week[column], label=column.replace("_", " "))
    ax.set_xlim(first, last); ax.set_ylim(0, 1000)
    ax.set_xticks(pd.date_range(first, last, freq="D")); ax.set_xticklabels([t.strftime("%a %d %b") for t in pd.date_range(first, last, freq="D")])
    ax.set_ylabel(unit); ax.set_title(title); ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

def plot_yearly_radiation(frame):
    # Total radiation of each year, in kWh per square metre.
    yearly = frame["shortwave_radiation"].resample("YE").sum() / 1000
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(yearly.index.year, yearly.values, color="#f4a261")
    for year, value in zip(yearly.index.year, yearly.values):
        ax.text(year, value + 10, f"{value:.0f}", ha="center", fontsize=8)
    ax.set_ylabel("kWh / m² per year"); ax.set_title("Solar radiation on a horizontal surface in Dingle, by year")
    ax.set_ylim(0, yearly.max() * 1.15); plt.tight_layout(); plt.show()

def make_windows(frame, inputs, window):
    # Cuts the hourly table into windows. Each window holds the last `window` hours of the
    # chosen inputs, and its target is the radiation of the next hour. Every column is divided
    # by its largest value of the training years, so that all the inputs sit between 0 and 1.
    X = (frame[inputs] / SCALE[inputs]).values.astype(np.float32)
    y = (frame["shortwave_radiation"] / SCALE["shortwave_radiation"]).values.astype(np.float32)
    Xw = np.lib.stride_tricks.sliding_window_view(X, window, axis=0)       # (n - window + 1, inputs, window)
    Xw = np.ascontiguousarray(Xw[:-1].transpose(0, 2, 1))                   # (n - window, window, inputs)
    yw = y[window:]
    times = frame.index[window:]
    return torch.tensor(Xw), torch.tensor(yw), times

def select_year(X, y, times, first_year, last_year=None):
    last_year = last_year or first_year
    keep = (times.year >= first_year) & (times.year <= last_year)
    return X[keep], y[keep], times[keep]

def predict(model, X, batch_size=4096):
    # Runs the model over many windows and returns the forecast in W/m².
    model.eval(); out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            out.append(model(X[start:start + batch_size]).squeeze(1))
    return torch.cat(out).numpy() * SCALE["shortwave_radiation"]

def mae(forecast, truth):
    # Mean absolute error: the average distance between the forecast and the truth, in W/m².
    return float(np.mean(np.abs(np.asarray(forecast) - np.asarray(truth))))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def plot_run(history, forecast_week, truth_week, title):
    # Left: the loss after each epoch. Right: one week of the validation year, truth and forecast.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.2), gridspec_kw={"width_ratios": [1, 2.2]})
    epochs = range(1, len(history) + 1)
    ax1.plot(epochs, history, marker="o"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.set_title("Loss on the training years"); ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.plot(truth_week.index, truth_week.values, label="radiation (Open-Meteo)", color="#444")
    ax2.plot(forecast_week.index, forecast_week.values, label="forecast of the model", color="#e76f51")
    ax2.set_ylabel("W/m²"); ax2.set_title(f"One week of {VALID_YEAR}"); ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3); fig.suptitle(title, fontsize=10); plt.tight_layout(); plt.show()

def show_log():
    # Prints the experiment log as a table, with the "beats yesterday" column of the challenge.
    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("model", "inputs")} for r in experiment_log])
    table["target reached"] = np.where((table["valid_mae"] < TARGET_MAE) & (table["seconds"] <= 60), "yes", "")
    print(table.to_string(index=False))

def energy_from_radiation(radiation, system_kw, tilt_factor, performance_factor):
    # The formula of Part 0 of the practical page, for a series of hourly radiation in W/m².
    # radiation / 1000 is kWh per square metre in that hour. The tilt factor converts a horizontal
    # surface to the roof. The system power and the performance factor give kWh from the panels.
    return radiation.clip(lower=0) / 1000 * tilt_factor * system_kw * performance_factor

def consumption_profile(house):
    # The typical consumption of the family for each month and each hour of the day, in kWh,
    # from their records of the real year. Consumption = bought + produced - sent to the grid.
    # The records have gaps in some months. A month without records takes the profile of the whole year.
    rows = house[house.index.year == REAL_YEAR]
    consumption = (rows["import_kWh"] + rows["solar_kWh"] - rows["export_kWh"]).clip(lower=0).resample("h").sum(min_count=2)
    by_month_hour = consumption.groupby([consumption.index.month, consumption.index.hour]).mean()
    by_hour = consumption.groupby(consumption.index.hour).mean()
    profile = pd.Series({(m, h): by_month_hour.get((m, h), by_hour[h]) for m in range(1, 13) for h in range(24)})
    return profile.fillna(by_hour.mean())

def typical_consumption(profile, index):
    # The profile laid over the hours of a given year.
    return pd.Series([profile.loc[(t.month, t.hour)] for t in index], index=index)

def savings(energy, consumption, buy_price, export_rate):
    # Splits the energy of each hour into the part used at home and the part sent to the grid,
    # and values each part at its price.
    used_at_home = np.minimum(energy, consumption)
    exported = energy - used_at_home
    return {"energy_kWh": float(energy.sum()), "used_at_home_kWh": float(used_at_home.sum()),
            "exported_kWh": float(exported.sum()),
            "saving_eur": float(used_at_home.sum() * buy_price + exported.sum() * export_rate)}

def plot_money(lines, net_cost, years):
    # The money line: minus the net cost at year 0, plus the yearly saving each year.
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for label, saving, colour in lines:
        balance = [-net_cost + saving * y for y in range(years + 1)]
        payback = net_cost / saving if saving > 0 else float("inf")
        ax.plot(range(years + 1), balance, marker="o", markersize=3, color=colour, label=f"{label}: payback {payback:.1f} years")
    ax.axhline(0, color="gray", linewidth=1); ax.set_xlabel("years after the installation"); ax.set_ylabel("euro")
    ax.set_title("Money balance of the family"); ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

# ---- Two building blocks for the model ---------------------------------------
# PyTorch's nn.LSTM returns two things at once, so it cannot sit in an nn.Sequential chain as it is.
# LSTMBlock wraps it and returns only the memory after each hour. LastHour keeps the memory of the
# last hour of the window, so that a Linear layer can read it. With these two blocks, the model is a
# chain of layers like the models of Practical 5.

class LSTMBlock(nn.Module):
    # Reads the window one hour at a time and keeps a memory of `units` numbers.
    def __init__(self, n_inputs, units):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, units, batch_first=True)
    def forward(self, x):                      # x: windows x hours x inputs
        memory, _ = self.lstm(x)               # memory after each hour: windows x hours x units
        return memory

class LastHour(nn.Module):
    # Keeps only the memory after the last hour of the window: windows x units.
    def forward(self, x):
        return x[:, -1, :]

print(f"Setup done. The family of the session is house {HOUSE}.")
""", title="Setup: libraries, data and helper functions (about 20 seconds)", hidden=True)

# =========================================================================== Part 1
md(r"""
## Part 1 — The solar radiation at the house

### 1.1 Loading the weather

The weather file holds one row per hour from 2010 to 2021 for the coordinates of Dingle (52.14 N, 10.27 W). The notebook keeps the years up to 2019. The cell loads the file, prints the first rows and the columns, and draws three figures. The first figure is the total radiation of each year. The other two show one week of June and one week of December, so you can see the shape of the daily curve and how much it changes with the season and the clouds.

The column `shortwave_radiation` is the **global horizontal radiation**, i.e., all the solar energy that reaches one square metre of flat ground in one hour, in W/m². It is the variable that the model forecasts. The other columns are the candidates for the inputs of the challenge.
""")
code(r"""
weather = load_weather()
print("Rows:", len(weather), "   Columns:", list(weather.columns))
print(weather.loc["2019-06-21 10:00":"2019-06-21 13:00"].round(1).to_string())

plot_yearly_radiation(weather)
plot_week(weather, "2019-06-17", ["shortwave_radiation", "direct_radiation", "diffuse_radiation"], "A week of June 2019", "W/m²")
plot_week(weather, "2019-12-09", ["shortwave_radiation", "direct_radiation", "diffuse_radiation"], "A week of December 2019", "W/m²")
""", title="Load the weather and draw a year, a week of June and a week of December", hidden=True)

md(r"""
### 1.2 The split in time

The model learns from the years 2010 to 2017. During the challenge, it is scored on 2018, a year it has never seen. The year 2019 stays aside for the estimate of Part 4 and the comparison of Part 5. A split in time is the honest way to test a forecast. A model that is scored on hours it has seen during training looks better than it is.

The cell also computes the **scale** of each column, i.e., its largest value in the training years. Every input is divided by its scale before it enters the model, so that all the inputs sit between 0 and 1.
""")
code(r"""
train = weather[(weather.index.year >= TRAIN_YEARS[0]) & (weather.index.year <= TRAIN_YEARS[1])]
SCALE = train.max()                       # the largest value of each column in the training years
print(f"Training years {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}: {len(train):,} hours")
print(f"Validation year {VALID_YEAR}:        {(weather.index.year == VALID_YEAR).sum():,} hours")
print(f"Real year {REAL_YEAR}:              {(weather.index.year == REAL_YEAR).sum():,} hours (kept aside)")
print("\nScale of each column (largest value in the training years):")
print(SCALE.round(1).to_string())
""", title="Split the years and compute the scale of each column", hidden=True)

# =========================================================================== Part 2
md(r"""
## Part 2 — Simple guesses and a first model

### 2.1 Two simple guesses

Before you train anything, measure two guesses that need no model. The first guess says that the radiation of the next hour equals the radiation of this hour. The second guess says that it equals the radiation of the same hour yesterday. The cell scores both guesses on the validation year with the **mean absolute error (MAE)**, i.e., the average distance between the guess and the truth over all the hours of the year, in W/m². A model is only useful when it beats these two numbers.
""")
code(r"""
valid = weather[weather.index.year == VALID_YEAR]["shortwave_radiation"]
baseline_last_hour = mae(valid.shift(1).iloc[24:], valid.iloc[24:])
baseline_yesterday = mae(valid.shift(24).iloc[24:], valid.iloc[24:])
print(f"MAE of 'the same as this hour':          {baseline_last_hour:5.1f} W/m²")
print(f"MAE of 'the same as this hour yesterday': {baseline_yesterday:5.1f} W/m²")
""", title="Score the two simple guesses on the validation year", hidden=True)

md(r"""
### 2.2 The training loop

The cell below defines the two functions that you call in the challenge. You need to know what each function does, because the challenge asks you to read what they print. Double-click the title if you want to read the code.

- `train_model` runs the gradient steps for a number of epochs and records the loss after each epoch. The loss is the squared distance between the forecast and the truth, averaged over the windows of a batch.
- `run_experiment` cuts the hours into windows with the inputs you chose, and calls `train_model` on them from fresh random parameters. Then, it scores the model on the validation year, draws the loss and one validation week, and adds one row to an **experiment log**. The log is a table with every model you have tried.

The window is the number of past hours that the model sees before it forecasts the next hour. With `window=24`, the model reads the last day.
""")
code(r"""
def train_model(model, X, y, epochs, learning_rate, batch_size):
    loss_function = nn.MSELoss()                                   # squared distance between forecast and truth
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X))                            # shuffle the windows
        losses = []
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_function(model(X[batch]).squeeze(1), y[batch])
            loss.backward()                                        # direction that lowers the loss, for each parameter
            optimizer.step()                                       # move every parameter one step
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
    return history

experiment_log = []      # one row per model you train

def run_experiment(name, model, inputs, window, epochs, learning_rate, batch_size=256, seed=0):
    torch.manual_seed(seed)
    for layer in model.modules():                                  # fresh random parameters for each run
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    X_all, y_all, t_all = make_windows(weather, inputs, window)
    X_train, y_train, _ = select_year(X_all, y_all, t_all, TRAIN_YEARS[0], TRAIN_YEARS[1])
    X_valid, y_valid, t_valid = select_year(X_all, y_all, t_all, VALID_YEAR)
    t0 = time.time()
    history = train_model(model, X_train, y_train, epochs, learning_rate, batch_size)
    seconds = time.time() - t0
    forecast = pd.Series(predict(model, X_valid), index=t_valid)
    truth = pd.Series(y_valid.numpy() * SCALE["shortwave_radiation"], index=t_valid)
    row = {"run": len(experiment_log) + 1, "name": name, "n_inputs": len(inputs), "window": window,
           "parameters": count_parameters(model), "epochs": epochs, "learning_rate": learning_rate,
           "batch_size": batch_size, "seconds": round(seconds, 1),
           "train_mae": round(mae(predict(model, X_train[-8760:]), y_train[-8760:].numpy() * SCALE["shortwave_radiation"]), 1),
           "valid_mae": round(mae(forecast, truth), 1), "model": model, "inputs": list(inputs)}
    experiment_log.append(row)
    week = slice(f"{VALID_YEAR}-06-11", f"{VALID_YEAR}-06-18")
    plot_run(history, forecast[week], truth[week], f"run {row['run']}: {name}  (validation MAE {row['valid_mae']} W/m²)")
    show_log()
    return model

print("Training functions ready.")
""", title="Training functions: train_model and run_experiment", hidden=True)

md(r"""
### 2.3 A first model

The cell defines the model and trains it. This is your starting point for the challenge. As in Practical 5, the model is a chain of layers inside `nn.Sequential`. The window enters the first layer, and the output of each layer feeds the next one. The first model has three layers.

- `LSTMBlock(1, 16)` is the **LSTM layer**. It reads the window one hour at a time. At each hour it receives the inputs of that hour and updates a memory of 16 numbers. The memory keeps what matters from the past hours and forgets the rest. The first argument is the number of inputs at each hour, and the second is the size of the memory, which we call the units.
- `LastHour()` keeps only the memory after the last hour of the window. The memory after the last hour summarises the whole window. This layer has no parameters.
- `nn.Linear(16, 1)` is the **linear layer** of Practical 5. It multiplies the 16 numbers of the memory by 16 weights, adds the products and a bias, and gives one number, the forecast of the next hour. Its first argument must equal the number of units of the block before it.

The first model reads only the radiation of the last 24 hours and trains for 5 epochs with a learning rate of 0.001. Run the cell and read the output. The right chart shows one week of the validation year, with the truth in grey and the forecast in orange. The table shows the validation MAE next to the two simple guesses.
""")
code(r"""
INPUTS = ["shortwave_radiation"]           # the columns that the model reads at each hour of the window
first_model = nn.Sequential(
    LSTMBlock(len(INPUTS), 16),            # reads the window hour by hour, memory of 16 numbers
    LastHour(),                            # keeps the memory after the last hour
    nn.Linear(16, 1),                      # 16 numbers x 16 weights -> the forecast
)
print("Parameters of the first model:", count_parameters(first_model))
run_experiment("first model: radiation only", first_model, INPUTS, window=24, epochs=5, learning_rate=0.001)
""")
md(r"""
The first model beats the two guesses. The loss is still going down when the training stops. The challenge starts here.
""")

# =========================================================================== Part 3
md(r"""
## Part 3 — The challenge: forecast the radiation better

Your goal is a model with a **validation MAE below 18 W/m²**, trained in **at most 60 seconds**. Among the models that reach the target, the one with the lowest MAE wins, and the fewest parameters break a tie. The log marks every run that meets the two conditions with "yes" in the last column. When you find a good configuration, run it again with `seed=1`. The MAE moves by one or two W/m² from one seed to the next, so a good configuration reaches the target with both seeds.

### What you can change

The table lists every factor of the model and the training, what it does, and what to expect when you change it.

| Factor | What it does | What to expect |
|---|---|---|
| `INPUTS` (list at the top of the cell) | the columns the model reads at each hour | more inputs give more to work with, e.g., the clouds say whether the sky is clearing |
| `window` (argument of `run_experiment`) | how many past hours the model reads | 12 is fast, 24 is one day, 48 is two days and twice the time |
| units of `LSTMBlock(n_inputs, units)` | the size of the memory | 8 is small and fast, 16 is the default, 32 or 64 remember more and take longer |
| a second `LSTMBlock(units, units2)` after the first | a second memory that reads the first one | a little more accuracy at twice the time, its first argument = the units before |
| `nn.Linear(units, hidden)` + `nn.ReLU()` after `LastHour()` | a hidden layer before the forecast | helps when the memory is large, the next `nn.Linear` starts with `hidden` |
| `nn.Dropout(0.1)` between layers | switches off 10% of the numbers during training | less dependence on any single number, training MAE may rise a little |
| `epochs` | passes of the training over all the windows | more epochs lower the loss until it flattens, about one second each |
| `learning_rate` | the size of each gradient step | 0.001 is slow and safe, 0.003 is a good default, 0.01 may jump around |
| `batch_size` | windows seen before each step | 256 is the default, 64 gives more steps and takes longer |
| `seed` | the random start of the parameters | moves the MAE by one or two W/m², so check a good configuration with two seeds |

**The menu of inputs.** Add a column by adding its name to the list `INPUTS`. The first argument of the first `LSTMBlock` is `len(INPUTS)`, which the cell sets for you.

| Column | What it is |
|---|---|
| `shortwave_radiation` | global radiation on a horizontal surface, W/m² |
| `direct_radiation` | the part of the radiation that comes straight from the sun, W/m² |
| `diffuse_radiation` | the part scattered by the sky and the clouds, W/m² |
| `direct_normal_irradiance` | direct radiation on a surface that faces the sun, W/m² |
| `global_tilted_irradiance` | global radiation on a roof tilted 30° towards the south, W/m² |
| `cloud_cover`, `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high` | share of the sky covered by clouds, in total and by height, % |
| `temperature_2m` | air temperature, °C |
| `relative_humidity_2m` | relative humidity, % |
| `wind_speed_10m` | wind speed, m/s |
| `hour_of_day` | 0 to 23 |
| `day_of_year` | 1 to 366 |

**How to work.**

1. Copy the cell below (or edit it in place), change **one thing**, give the run a short name that says what you changed, and run it. Every run adds a row to the log, so you can always compare.
2. Look at the loss before you decide the next change. A loss that is still going down asks for more epochs. A loss that jumps up and down asks for a smaller learning rate. A validation MAE that gets worse when the training MAE gets better means that the model memorises the training years.
3. Look at the week chart. A forecast that lags behind the truth by one hour is a model that copies the last hour. The inputs about the clouds help the model to anticipate a change.
4. The rule of shape is the one of Practical 5. The first argument of each layer must equal the number of numbers that arrive from the layer before. If you get it wrong, the error message tells you the two numbers that did not match.

Keep each training under a minute. A model with 16 units and 10 epochs takes about 15 seconds, so you have time for many runs.
""")
code(r"""
# Edit this cell. Change one thing at a time and give each run a name that says what you changed.
INPUTS = ["shortwave_radiation"]           # add columns from the menu, e.g., "cloud_cover", "direct_radiation"
my_model = nn.Sequential(
    LSTMBlock(len(INPUTS), 16),            # memory of 16 numbers   (try 8, 32, or a second LSTMBlock(16, 16) after this one)
    LastHour(),                            # keeps the memory after the last hour
    nn.Linear(16, 1),                      # 16 = the units of the block before   (try nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
)
run_experiment("attempt 1: same as first model", my_model, INPUTS, window=24, epochs=5, learning_rate=0.001, batch_size=256, seed=0)
""")
if TEACHER:
    md(r"""
**Teacher notes — a worked path through the challenge.** The cells below show one sequence of edits with the results observed on a 2-core CPU (seed 0). The learning rate and the epochs come first. Ten epochs at 0.003 take the first model from about 23 to about 20 W/m². Then, the cloud cover takes it to about 18, on the edge of the target, and the components of the radiation with the temperature, the humidity and the wind take it to about 17, below the target. The hour of the day and the day of the year add little, because the window already tells the model where it stands in the day. The MAE moves by one or two W/m² between seeds. A second block, a hidden layer or more units give little and cost time. Eight units stay above the target.
""")
    code(r"""
INPUTS = ["shortwave_radiation"]
my_model = nn.Sequential(LSTMBlock(len(INPUTS), 16), LastHour(), nn.Linear(16, 1))
run_experiment("attempt 2: 10 epochs, learning rate 0.003", my_model, INPUTS, window=24, epochs=10, learning_rate=0.003)
""", title="Teacher: attempt 2 — more epochs and a larger learning rate (about 20 W/m²)")
    code(r"""
INPUTS = ["shortwave_radiation", "cloud_cover"]
my_model = nn.Sequential(LSTMBlock(len(INPUTS), 16), LastHour(), nn.Linear(16, 1))
run_experiment("attempt 3: + cloud cover", my_model, INPUTS, window=24, epochs=10, learning_rate=0.003)
""", title="Teacher: attempt 3 — add the cloud cover (about 18 W/m², on the edge of the target)")
    code(r"""
INPUTS = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "cloud_cover", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
my_model = nn.Sequential(LSTMBlock(len(INPUTS), 16), LastHour(), nn.Linear(16, 1))
run_experiment("attempt 4: + direct, diffuse, temperature, humidity, wind", my_model, INPUTS, window=24, epochs=10, learning_rate=0.003)
run_experiment("attempt 5: same, seed 1", my_model, INPUTS, window=24, epochs=10, learning_rate=0.003, seed=1)
""", title="Teacher: attempts 4 and 5 — seven inputs, two seeds (about 17 W/m²)")
    code(r"""
my_model = nn.Sequential(
    LSTMBlock(len(INPUTS), 32),
    LSTMBlock(32, 32),                     # a second memory that reads the first one
    LastHour(),
    nn.Dropout(0.1),
    nn.Linear(32, 8), nn.ReLU(),           # a hidden layer
    nn.Linear(8, 1),
)
run_experiment("attempt 6: two blocks of 32, dropout, hidden layer", my_model, INPUTS, window=24, epochs=10, learning_rate=0.003)
""", title="Teacher: attempt 6 — a bigger chain (little gain, about 50 seconds)")
    code(r"""
my_model = nn.Sequential(LSTMBlock(len(INPUTS), 8), LastHour(), nn.Linear(8, 1))
run_experiment("attempt 7: 8 units", my_model, INPUTS, window=24, epochs=10, learning_rate=0.003)
""", title="Teacher: attempt 7 — 8 units (fast, but above the target)")

md(r"""
**Two examples of valid edits**, to show the rule of shape at work. Do not run them as they are. Use them as a guide.

A second LSTM block and more inputs:

```python
INPUTS = ["shortwave_radiation", "cloud_cover", "temperature_2m"]
my_model = nn.Sequential(
    LSTMBlock(len(INPUTS), 16),            # 3 inputs -> memory of 16
    LSTMBlock(16, 32),                     # reads the 16 memories -> memory of 32   (in = 16, the units before)
    LastHour(),
    nn.Linear(32, 1),                      # 32 = the units of the block before
)
run_experiment("attempt 2: two blocks, three inputs", my_model, INPUTS, window=24, epochs=10, learning_rate=0.003)
```

A hidden linear layer with dropout before the forecast:

```python
my_model = nn.Sequential(
    LSTMBlock(len(INPUTS), 32),
    LastHour(),
    nn.Dropout(0.1),                       # switches off 10% of the 32 numbers during training
    nn.Linear(32, 8),                      # hidden layer: 32 -> 8
    nn.ReLU(),
    nn.Linear(8, 1),                       # 8 -> the forecast
)
run_experiment("attempt 3: hidden layer and dropout", my_model, INPUTS, window=48, epochs=10, learning_rate=0.003)
```
""")

# =========================================================================== Part 4
md(r"""
## Part 4 — Monetary projection

### 4.1 From radiation to electricity

The cell picks the model with the lowest validation MAE from your log and runs it over the year 2019, hour by hour. Then, it applies the formula of Part 0 of the practical page to the forecast. Three numbers enter the formula.

- `SYSTEM_KW`, the power of the panels. The four families installed 2.1 kW.
- `TILT_FACTOR`, how much more radiation a roof tilted 30° towards the south receives than a flat surface over a year. The cell computes it from the training years of the weather file, i.e., the ratio between `global_tilted_irradiance` and `shortwave_radiation`.
- `PERFORMANCE_FACTOR`, the share of the energy that the panels deliver after all the losses, i.e., dirt, temperature, cables and the inverter. A common value for a new system is 0.80.

The cell prints the energy per month and per year.
""")
code(r"""
best = min(experiment_log, key=lambda r: r["valid_mae"])
print(f"Best model: run {best['run']} '{best['name']}', validation MAE {best['valid_mae']} W/m²")

X_all, y_all, t_all = make_windows(weather, best["inputs"], best["window"])
X_real, y_real, t_real = select_year(X_all, y_all, t_all, REAL_YEAR)
radiation_estimate = pd.Series(predict(best["model"], X_real), index=t_real).clip(lower=0)

SYSTEM_KW = 2.1
TILT_FACTOR = float(train["global_tilted_irradiance"].sum() / train["shortwave_radiation"].sum())
PERFORMANCE_FACTOR = 0.80
energy_estimate = energy_from_radiation(radiation_estimate, SYSTEM_KW, TILT_FACTOR, PERFORMANCE_FACTOR)

print(f"Tilt factor of a 30° south-facing roof in Dingle: {TILT_FACTOR:.2f}")
monthly = energy_estimate.resample("ME").sum().rename(lambda t: t.strftime("%b")).rename_axis(None)
print(f"\nEstimated electricity of the panels in {REAL_YEAR}, kWh per month:")
print(monthly.round(0).to_string())
print(f"\nEstimated electricity in {REAL_YEAR}: {energy_estimate.sum():.0f} kWh")
""", title="Run the best model over 2019 and convert the radiation into electricity", hidden=True)

md(r"""
### 4.2 From electricity to savings

The electricity of the panels has two destinations. The part that the family uses at the moment it is produced replaces electricity that they would have bought, so it is worth the **buying price**. The part that nobody uses at that moment goes to the grid, and the supplier pays the **export rate** for it, which is lower. Therefore, the saving depends on when the family consumes.

The cell builds the typical consumption of the family for each month and each hour of the day from their records, lays it over the hours of 2019, and splits the estimated energy of each hour into the two destinations. The prices are the ones of the Irish market in November 2026. The practical page gives their sources.
""")
code(r"""
BUY_PRICE = 0.35          # euro per kWh bought from the grid (typical unit rate, November 2026)
EXPORT_RATE = 0.195       # euro per kWh sent to the grid (Clean Export Guarantee of the main suppliers, November 2026)

house = load_house(HOUSE)
profile = consumption_profile(house)
consumption = typical_consumption(profile, energy_estimate.index)
print(f"Typical consumption of house {HOUSE}: {consumption.sum():.0f} kWh per year, "
      f"{consumption.groupby(consumption.index.hour).mean().idxmax()}:00 is the hour of the largest use")

projection = savings(energy_estimate, consumption, BUY_PRICE, EXPORT_RATE)
print(f"\nProjection for {REAL_YEAR}")
print(f"  electricity of the panels: {projection['energy_kWh']:7.0f} kWh")
print(f"  used at home:              {projection['used_at_home_kWh']:7.0f} kWh  x {BUY_PRICE:.3f} €/kWh")
print(f"  sent to the grid:          {projection['exported_kWh']:7.0f} kWh  x {EXPORT_RATE:.3f} €/kWh")
print(f"  saving of the family:      {projection['saving_eur']:7.0f} € per year")

fig, ax = plt.subplots(figsize=(8, 3))
june = slice(f"{REAL_YEAR}-06-01", f"{REAL_YEAR}-06-30")
ax.plot(range(24), energy_estimate[june].groupby(energy_estimate[june].index.hour).mean(), label="panels (estimate)", color="#f4a261")
ax.plot(range(24), consumption[june].groupby(consumption[june].index.hour).mean(), label="consumption of the family", color="#264653")
ax.set_xlabel("hour of the day"); ax.set_ylabel("kWh per hour"); ax.set_title(f"A typical day of June {REAL_YEAR}, house {HOUSE}")
ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
""", title="Typical consumption of the family and the saving of the projection", hidden=True)

md(r"""
### 4.3 The payback

The family pays the system and receives the grant of the Sustainable Energy Authority of Ireland (SEAI). The net cost is what they pay. The payback is the net cost divided by the yearly saving. The cell draws the money balance of the family over 20 years. It starts at minus the net cost and climbs by the saving each year. The year in which the line crosses zero is the payback.

The figures below are the ones of November 2026. Edit them and run the cell again. Then, try a bigger system, a smaller export rate, or a higher electricity price, and watch the crossing move.
""")
code(r"""
COST_PER_KW = 2100        # euro per kW installed (SEAI gives a typical range of 1,700 to 2,500)
GRANT = 1800              # SEAI grant for a 2 kW system (900 euro per kW up to 2 kW)
YEARS = 20

net_cost = SYSTEM_KW * COST_PER_KW - GRANT
payback = net_cost / projection["saving_eur"]
print(f"System: {SYSTEM_KW} kW x {COST_PER_KW} € = {SYSTEM_KW * COST_PER_KW:.0f} €, minus the grant of {GRANT} € = net cost {net_cost:.0f} €")
print(f"Saving of the projection: {projection['saving_eur']:.0f} € per year  ->  payback {payback:.1f} years")
plot_money([("projection", projection["saving_eur"], "#e76f51")], net_cost, YEARS)
""")

# =========================================================================== Part 5
md(r"""
## Part 5 — The reality

### 5.1 What the panels really produced

The family installed its panels and measured them every 30 minutes. The cell loads the real production of the family in 2019 and draws it next to your estimate, month by month. Then, it prints the real energy, the real saving and the gap in percent. A month in which the records cover less than 90% of the half hours is marked, and the yearly totals only count the complete months, for both the estimate and the reality.

Write your gap on the board, next to the gaps of the other teams. The family is the same for everyone, so the differences between the gaps come from the models. Then, look for the causes of the gap. The practical page lists the usual ones.
""")
code(r"""
all_half_hours = pd.date_range(f"{REAL_YEAR}-01-01", f"{REAL_YEAR}-12-31 23:30", freq="30min")
real_solar = house["solar_kWh"].reindex(all_half_hours).clip(upper=SYSTEM_KW / 2)   # a half hour cannot exceed 1.05 kWh
real_energy = real_solar.resample("h").sum(min_count=1)                             # kWh per hour
counts = real_solar.resample("ME").count()
coverage = counts / (counts.index.days_in_month * 48)                                # share of the half hours with a record
complete = (coverage >= 0.90).values

real_monthly = real_energy.resample("ME").sum()
est_monthly = energy_estimate.resample("ME").sum()
table = pd.DataFrame({"estimate_kWh": est_monthly.round(0).values, "real_kWh": real_monthly.round(0).values,
                      "records": [f"{c * 100:.0f} %" for c in coverage.values],
                      "complete": np.where(complete, "yes", "no")}, index=est_monthly.index)
table.index = table.index.strftime("%b")
print(table.rename_axis(None).to_string())

est_total = est_monthly[complete].sum(); real_total = real_monthly[complete].sum()
gap = (est_total - real_total) / real_total * 100
print(f"\nOver the {int(complete.sum())} complete months of {REAL_YEAR}:")
print(f"  estimate: {est_total:6.0f} kWh     reality: {real_total:6.0f} kWh     gap: {gap:+.0f} %")

reality = savings(real_energy.reindex(energy_estimate.index).fillna(0), consumption, BUY_PRICE, EXPORT_RATE)
print(f"  real saving of the family: {reality['saving_eur']:.0f} € per year   (projection: {projection['saving_eur']:.0f} €)")

fig, ax = plt.subplots(figsize=(9, 3.2))
x = np.arange(12)
ax.bar(x - 0.2, est_monthly.values, 0.4, label="estimate", color="#e76f51")
ax.bar(x + 0.2, real_monthly.values, 0.4, label="reality", color="#2a9d8f")
for i, ok in enumerate(complete):
    if not ok:
        ax.text(i + 0.2, real_monthly.values[i] + 5, "incomplete", ha="center", fontsize=7, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(table.index); ax.set_ylabel("kWh per month")
ax.set_title(f"House {HOUSE}, {REAL_YEAR}: estimate of the engineer and measurement of the panels")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); plt.tight_layout(); plt.show()
""", title="Compare the estimate with the real production, month by month", hidden=True)

md(r"""
### 5.2 The payback, again

The cell draws the money balance twice, with the saving of your projection and with the real saving. The distance between the two payback years is the cost of the gap for the family.
""")
code(r"""
print(f"Net cost {net_cost:.0f} €.  Projection: {projection['saving_eur']:.0f} € per year.  Reality: {reality['saving_eur']:.0f} € per year.")
plot_money([("projection", projection["saving_eur"], "#e76f51"), ("reality", reality["saving_eur"], "#2a9d8f")], net_cost, YEARS)
""", title="The money balance with the projection and with the reality", hidden=True)

md(r"""
**Something went wrong?**

- If the setup cell cannot download the data files, run it again. If the error repeats, download the two files from the practical page, upload them in the Files panel (folder icon on the left), and run the setup cell once more.
- `RuntimeError: input.size(-1) must be equal to input_size`. The first `LSTMBlock` expects a different number of inputs than the list `INPUTS` holds. Keep `LSTMBlock(len(INPUTS), ...)` as the first layer and run the cell again after you change the list.
- `RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x32 and 16x1)`. A `nn.Linear` expects 16 numbers but receives 32. Set its first argument to the units of the layer before it.
- `KeyError: 'cloud'`. A name in `INPUTS` is not a column of the weather table. Copy the names from the table of Part 3.
- If a run takes more than a minute, reduce the epochs, the units or the window.
- If the loss shows `nan`, lower the learning rate.
- Part 4 needs at least one run in the log. If you restarted Colab, run the cells of Parts 1 to 3 again.
""")

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 0,
}
out_name = "MEEN41490_P6_Solar_Savings_TEACHER.ipynb" if TEACHER else "MEEN41490_P6_Solar_Savings_STUDENT.ipynb"
with open(out_name, "w") as f:
    json.dump(nb, f, indent=1)
print("written", len(cells), "cells ->", out_name)
