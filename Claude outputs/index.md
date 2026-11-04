---
layout: default
title: "Practical 6 — Forecasting solar radiation with an LSTM to estimate the savings and payback of rooftop panels"
---

# Practical 6 — Forecasting solar radiation with an LSTM to estimate the savings and payback of rooftop panels

## How this session works

**In this session, you work as the engineer of a family that wants solar panels.** Four families in Dingle, Co. Kerry, pay high electricity bills and want panels on their roofs to pay less. Each family asks its engineer two questions. How much electricity would the panels produce in a year? And in how many years do we recover what we pay for them? Each team takes one family and answers both questions with a notebook.

| Part | What you do |
|---|---|
| 0 — Read on this page | Where the data come from, the curve of a panel, the curve of a house, and the formula from radiation to electricity |
| 1 to 3 — Notebook | Train a model that forecasts the solar radiation at the house, then improve it in the challenge |
| 4 — Notebook | Turn the forecast into electricity, into savings, and into a payback time |
| 5 — Notebook | Compare your estimate with what the panels of your family really produced |

In Practical 5, you trained a model that looked at an image and answered yes or no. In this session, you train a model that looks at the **last hours** of the weather and answers a **number**, the solar radiation of the next hour. The model is a Long Short-Term Memory network (LSTM), a neural network with a small memory. The notebook keeps the same working pattern as Practical 5. The setup cells are collapsed, the training function keeps an experiment log, and in the challenge you edit the inputs and the model directly in code.

## Part 0 — What you need before the notebook

### 0.1 The real study behind the exercise

This practical is based on a real study. Between 2019 and 2022, four households on the Dingle Peninsula took part in a pilot of ESB Networks, the company that runs the electricity distribution network in Ireland. Each house received a rooftop solar system of 2.1 kW, a home battery, an electric car, a charger and a heat pump. Meters recorded what each house produced, bought and sent to the grid every 30 minutes for three years. The data were published in 2026 by Dimitrios Avgoloupis, Vasilis Michalakopoulos, Elissaios Sarmas and Vangelis Marinakis of the National Technical University of Athens, with Mario Couto, Alessio Coccia and Mariana Jimenez of EPRI Europe in Dublin.

> Avgoloupis, D., Michalakopoulos, V., Sarmas, E., Marinakis, V., Couto, M., Coccia, A. and Jimenez, M. (2026). *Residential Energy Dataset with Electric Vehicles, Photovoltaic Generation and Tariff Variability in Ireland.* Scientific Data, 13, 834. [https://doi.org/10.1038/s41597-026-07186-3](https://doi.org/10.1038/s41597-026-07186-3). Data on Figshare, [https://doi.org/10.6084/m9.figshare.30294478](https://doi.org/10.6084/m9.figshare.30294478), licence CC BY 4.0.

In the story of this session, the four households are the families that want the panels. Their records of consumption are what a real family would hand to an engineer, i.e., their bills. Their measured production is what you compare your estimate with in Part 5.

### 0.2 Where the solar radiation comes from

Nobody stands on the roof with an instrument. The solar radiation at the house comes from **satellites and weather models**. Weather satellites photograph the Earth every few minutes and measure the clouds and the light that the ground reflects. A weather model combines those images with the measurements of ground stations and with the physics of the atmosphere. The result is a reconstruction of the weather at every point of the planet, hour by hour, back to 1940. The European Centre for Medium-Range Weather Forecasts (ECMWF) produces one such reconstruction, called ERA5, and the Copernicus programme of the European Union publishes it for free.

The notebook reads ERA5 through **Open-Meteo**, a free service that returns the hourly weather of any coordinates. For Dingle, 52.14 N and 10.27 W, the file holds twelve years of hours, with these columns.

| Column | What it is | Unit |
|---|---|---|
| `shortwave_radiation` | **global horizontal radiation**, all the solar energy that reaches one square metre of flat ground in that hour | W/m² |
| `direct_radiation` | the part that comes straight from the sun | W/m² |
| `diffuse_radiation` | the part scattered by the sky and the clouds | W/m² |
| `direct_normal_irradiance` | the direct radiation on a surface that faces the sun | W/m² |
| `global_tilted_irradiance` | the global radiation on a roof tilted 30° towards the south | W/m² |
| `cloud_cover`, `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high` | share of the sky covered by clouds, in total and by height | % |
| `temperature_2m` | air temperature two metres above the ground | °C |
| `relative_humidity_2m` | relative humidity | % |
| `wind_speed_10m` | wind speed ten metres above the ground | m/s |

A watt per square metre (W/m²) is a power. Over one hour, 1000 W/m² deliver 1 kWh of energy to one square metre. A clear summer noon in Dingle reaches 700 to 900 W/m². A cloudy winter noon stays under 100 W/m². These values come from a model, not from an instrument on the roof. For this reason, the estimate of the engineer and the measurement of the panels will not agree exactly. Part 5 is about that gap.

The figure below shows one week of June 2019 with the radiation of Open-Meteo in grey and the real production of the panels of house 1 in orange. The two curves rise and fall together, because the panels respond to the radiation. The shape of each day depends on the clouds.

![One week of June 2019 with two curves: the solar radiation reconstructed from satellites, in W per square metre, and the real production of the panels of house 1, in kW. The curves rise and fall together each day, with dents on the cloudy days]({{ '/practicals/solar_savings_lstm/assets/radiation_and_panel_week.png' | relative_url }})

### 0.3 The curve of a solar panel

A solar panel produces nothing at night. After sunrise, its output rises with the sun, peaks around solar noon, and falls to zero at sunset. On a clear day, the curve is a smooth bell. On a cloudy day, the clouds cut dents into the bell, and the peak may reach only a quarter of the clear-day value. Over the year, the bell is tall and wide in June and short and narrow in December, because the days are longer and the sun is higher in summer.

![Three panels: the production of a 2.1 kW roof on a clear day of June 2019, a smooth bell that peaks near 1.9 kW; the same roof on a cloudy day, a jagged curve under 0.5 kW; and the monthly production of the roof over 2019, from under 40 kWh in January and December to about 300 kWh in June]({{ '/practicals/solar_savings_lstm/assets/curve_of_a_panel.png' | relative_url }})

The figure uses the real measurements of house 1. On 18 June 2019, a clear day, the 2.1 kW system peaked near 1.9 kW. On 23 June, a cloudy day, it stayed under 0.5 kW. Over the year 2019, the same roof produced 1,881 kWh, of which 297 kWh in June and 17 kWh in January. A rule of thumb for Ireland is 850 to 950 kWh per year for each kW of panels.

### 0.4 The curve of a house

A house follows a different clock. Consumption is low at night, rises in the morning when people get up, drops in the middle of the day when they are out, and peaks in the evening when they cook, wash and heat. Winter days consume more than summer days. The left panel of the figure shows the typical day of house 1 in December and in June.

![Two panels: the typical daily consumption of house 1 in December and in June, with a morning bump and an evening peak; and the June curve of the family next to the June curve of the panels, with the overlap of the two shaded as the part used at home]({{ '/practicals/solar_savings_lstm/assets/curve_of_a_house.png' | relative_url }})

The right panel puts the two curves of June on the same axes. **The panels produce most at midday, and the family consumes most in the morning and the evening.** The shaded area is the electricity that the family uses at the moment the panels produce it. The rest of the production goes to the grid. This mismatch drives the whole money calculation of Part 4.

- Electricity used at home replaces electricity that the family would have bought. It is worth the **buying price**, about 35 cent per kWh in November 2026.
- Electricity sent to the grid is paid by the supplier at the **export rate**, about 19.5 cent per kWh in November 2026. It is worth less.

Therefore, two families with the same roof save different amounts of money. The family that is at home at midday saves more.

### 0.5 The data, with real values

The notebook uses two kinds of files. The table below shows four rows of the file of house 1 on 21 June 2019, around midday.

| `timestamp` | `solar_kWh` | `import_kWh` | `export_kWh` |
|---|---|---|---|
| 2019-06-21 11:00 | 0.880 | 0.066 | 0.257 |
| 2019-06-21 11:30 | 0.924 | 0.046 | 0.528 |
| 2019-06-21 12:00 | 0.927 | 0.060 | 0.590 |
| 2019-06-21 12:30 | 0.886 | 0.000 | 0.694 |

- `timestamp` is the start of a 30-minute interval, in Coordinated Universal Time (UTC).
- `solar_kWh` is the electricity that the panels produced in that half hour, in kWh. A 2.1 kW system cannot produce more than 1.05 kWh in a half hour.
- `import_kWh` is the electricity that the family bought from the grid in that half hour.
- `export_kWh` is the electricity that the family sent to the grid in that half hour.

At 12:30 the panels produced 0.886 kWh, the family bought nothing and sent 0.694 kWh to the grid. Therefore, the family used 0.886 − 0.694 = 0.192 kWh of its own electricity in that half hour. In general, the consumption of the family in a half hour is `import_kWh + solar_kWh − export_kWh`. The notebook uses this sum to build the typical day of your family. The original file also holds the battery and the heat pump of the last months of the study, and a price scenario of the researchers. The notebook does not use them.

The second kind of file is the weather of Dingle. The table shows four rows of the same day.

| `time` | `shortwave_radiation` | `direct_radiation` | `diffuse_radiation` | `global_tilted_irradiance` | `cloud_cover` | `temperature_2m` | `wind_speed_10m` |
|---|---|---|---|---|---|---|---|
| 2019-06-21 10:00 | 601 | 430 | 171 | 614 | 100 | 14.7 | 2.6 |
| 2019-06-21 11:00 | 646 | 439 | 207 | 684 | 99 | 15.3 | 4.0 |
| 2019-06-21 12:00 | 735 | 561 | 174 | 806 | 81 | 15.1 | 4.7 |
| 2019-06-21 13:00 | 735 | 525 | 210 | 805 | 47 | 15.4 | 5.5 |

The radiation of each row is the average over the hour that ends at that time. At 12:00, 735 W/m² reached the flat ground, of which 561 came straight from the sun and 174 from the sky. A roof tilted 30° towards the south received 806 W/m², because it faces the sun better than flat ground.

### 0.6 From radiation to electricity

The engineer converts the radiation into the electricity of the panels with one formula.

> energy in one hour (kWh) = radiation (W/m²) / 1000 × tilt factor × system power (kW) × performance factor

- **radiation / 1000** is the energy that reaches one square metre of flat ground in that hour, in kWh.
- **The tilt factor** converts flat ground to the roof. A roof tilted 30° towards the south receives about 16% more radiation than flat ground over a year in Dingle. The notebook computes this factor from the weather file.
- **The system power**, 2.1 kW, is the power that the panels deliver under a radiation of 1000 W/m². Multiplying by it turns the radiation into the output of the panels.
- **The performance factor** accounts for all the losses, i.e., dirt on the panels, the heat of the panels, the cables and the inverter that converts the current for the house. A new system delivers about 80% of the ideal output, so the factor is 0.80.

For example, the hour of 12:00 on 21 June 2019 gives 735 / 1000 × 1.16 × 2.1 × 0.80 = 1.43 kWh. The real panels of house 1 produced 0.880 + 0.924 = 1.80 kWh in the two half hours of that hour, so the formula underestimates that particular hour. A cloudy winter hour of 50 W/m² gives 50 / 1000 × 1.16 × 2.1 × 0.80 = 0.10 kWh. The formula is simple and the engineers use richer versions of it. However, over a whole year, the simple version lands within a few percent of the measurement when the roof faces south and the system works without faults. You will see how close in Part 5.

### 0.7 Forecasting with a window and an LSTM

A **time series** is a list of measurements in time order, e.g., the radiation of every hour. A **forecast** uses the past values to estimate the next one. The model of this session reads a **window** of the last hours, e.g., the last 24 hours of radiation and clouds, and outputs one number, the radiation of the next hour. Then the window slides one hour forward, and the model forecasts the following hour.

The model is an LSTM. It reads the window one hour at a time and keeps a **memory** of a few numbers. At each hour, it decides what to add to the memory and what to forget. After the last hour of the window, the memory summarises the day, and a linear layer turns it into the forecast. A network that reads the whole window at once would need one weight for every hour and every input. The LSTM reuses the same weights at every hour, so it stays small, and its memory lets it notice a trend, e.g., clouds that have been clearing for three hours.

The score of a forecast is the **mean absolute error (MAE)**, i.e., the average distance between the forecast and the truth over all the hours of a year, in W/m². Two simple guesses give the numbers to beat. "The same as this hour" has an MAE of about 36 W/m² in Dingle. "The same as this hour yesterday" has an MAE of about 52 W/m². A model is only useful when it beats both.

## The notebook of the practice

The notebook has five parts, and the challenge sits in Part 3.

| Part | What you do | Time |
|---|---|---|
| 1 — The solar radiation at the house | Load the weather of Dingle, look at a year, a week of June and a week of December, split the years | 10 min |
| 2 — Simple guesses and a first model | Score the two simple guesses, meet the training loop, train the first LSTM | 15 min |
| 3 — The challenge | Edit the inputs and the model to reach the target MAE | 30 min |
| 4 — Monetary projection | Convert the forecast into electricity, savings and payback | 15 min |
| 5 — The reality | Compare the estimate with the real production of your family | 10 min |

Most cells are **collapsed**. A collapsed cell shows only its title, and you run it with the play button like any other cell. Double-click the title if you want to read the code. The cells that you edit in Parts 3 and 4 show their code.

> **Before you begin.** If Colab is new to you, read [Getting Started with Google Colab]({{ '/guides/getting-started-colab/' | relative_url }}) first. The notebook runs on the CPU, so accept the connection without GPU. Set `HOUSE` in the setup cell to the number of your family, from 1 to 4. Each team works with a different family.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/COLAB_LINK_PENDING)

### The data of the session

The setup cell downloads the five data files into Colab by itself. The links below are only for looking at the data.

- [`dingle_weather_2010_2021.csv`]({{ '/practicals/solar_savings_lstm/assets/data/dingle_weather_2010_2021.csv' | relative_url }}){:download=""}, the hourly weather of Dingle from Open-Meteo (ERA5, Copernicus Climate Change Service), 2010 to 2021, 5.9 MB.
- [`house1.csv`]({{ '/practicals/solar_savings_lstm/assets/data/house1.csv' | relative_url }}){:download=""}, [`house2.csv`]({{ '/practicals/solar_savings_lstm/assets/data/house2.csv' | relative_url }}){:download=""}, [`house3.csv`]({{ '/practicals/solar_savings_lstm/assets/data/house3.csv' | relative_url }}){:download=""}, [`house4.csv`]({{ '/practicals/solar_savings_lstm/assets/data/house4.csv' | relative_url }}){:download=""}, the records of the four families every 30 minutes, 2019 to 2022, 2.3 MB each, from Avgoloupis et al. (2026), CC BY 4.0.

## Part 1 — The solar radiation at the house

### 1.1 Loading the weather

The cell loads the weather file, prints the columns and four rows of June 2019, and draws three figures. The first figure is the total radiation of each year, in kWh per square metre. Dingle receives about 1,000 to 1,100 kWh/m² a year on flat ground, and the years differ by less than 10%. The other two figures show a week of June and a week of December, with the global, the direct and the diffuse radiation. Look at the height of the daily bells, at the hours of sunrise and sunset, and at what the clouds do to the direct part.

### 1.2 The split in time

The model learns from the years 2010 to 2017. During the challenge, it is scored on 2018, a year it has never seen. The year 2019 stays aside for the estimate of Part 4 and the comparison of Part 5. A split in time is the honest way to test a forecast. The cell also computes the **scale** of each column, i.e., its largest value in the training years. Every input is divided by its scale before it enters the model, so that all the inputs sit between 0 and 1. Without this step, the cloud cover, which goes up to 100, would weigh more than the temperature, which goes up to 28.

## Part 2 — Simple guesses and a first model

### 2.1 Two simple guesses

The cell scores "the same as this hour" and "the same as this hour yesterday" on 2018. Write the two numbers down. They are the first thing to beat.

### 2.2 The training loop

The cell defines `train_model` and `run_experiment`. The first one runs the gradient steps, as in Practical 5. The loss is now the squared distance between the forecast and the truth, because the answer is a number and not a label. The second one cuts the hours into windows with the inputs you chose and trains a model from fresh random parameters. Then, it scores the model on 2018, draws the loss and one week of June 2018 with the truth and the forecast, and adds one row to the experiment log.

### 2.3 A first model

The model has two parts. `nn.LSTM(n_inputs, units)` reads the window one hour at a time and keeps a memory of `units` numbers. `nn.Linear(units, 1)` turns the memory after the last hour into the forecast. The first model reads only the radiation of the last 24 hours, with 16 units, and trains for 5 epochs with a learning rate of 0.001. It has 1,233 parameters and trains in about ten seconds. It beats the two guesses, with an MAE near 23 W/m². The loss is still going down when the training stops.

## Part 3 — The challenge: forecast the radiation better

Your goal is a model with a **validation MAE of 20 W/m² or less**, trained in **at most 60 seconds**. Among the models that reach the target, the one with the lowest MAE wins, and the fewest parameters break a tie. The log marks every run that meets the two conditions with "yes" in the last column. When you find a good configuration, run it again with `seed=1`. The MAE moves by one or two W/m² from one seed to the next, so a good configuration reaches the target with both seeds.

You can change two kinds of things.

**The inputs.** The list `INPUTS` holds the columns that the model reads at each hour of the window. The menu is the table of Part 0.2, plus `hour_of_day` (0 to 23) and `day_of_year` (1 to 366). Add a column by adding its name to the list.

**The model and the training.** The window, the number of units, the number of layers, the dropout, the number of epochs, the learning rate and the batch size are the arguments of `RadiationModel` and `run_experiment`. Dropout switches off a share of the memory during training, which makes the model less dependent on any single number.

**How to work.** Change one thing at a time, give the run a name that says what you changed, and look at the loss and the week chart before the next change. A loss that is still going down asks for more epochs. A loss that jumps up and down asks for a smaller learning rate. A forecast that lags behind the truth by one hour is a model that copies the last hour. The inputs about the clouds help the model to anticipate a change. A model with 16 units and 10 epochs takes about 15 seconds, so you have time for many runs.

## Part 4 — Monetary projection

### 4.1 From radiation to electricity

The cell picks the model with the lowest validation MAE from your log and runs it over the year 2019, hour by hour. Then, it applies the formula of Part 0.6 with the system power of 2.1 kW, the tilt factor computed from the weather file, and a performance factor of 0.80. It prints the electricity of the panels per month and per year. Write the yearly number down. It is the first answer for your family.

### 4.2 From electricity to savings

The cell builds the typical day of your family for each month from their records, with the sum of Part 0.5. Then, it splits the electricity of each hour into the part used at home and the part sent to the grid, and values each part at its price. The prices are the ones of November 2026, and you can change them in the cell.

| Figure | Value | Source |
|---|---|---|
| Buying price of electricity | 0.35 € per kWh | typical unit rate of the Irish suppliers, November 2026 |
| Export rate | 0.195 € per kWh | Clean Export Guarantee of SSE Airtricity and Electric Ireland, November 2026, [Selectra](https://selectra.ie/energy/guides/prices/best-microgeneration-rates-ireland) |
| Cost of a system | 1,700 to 2,500 € per kW installed | [SEAI, Solar Electricity Grant](https://www.seai.ie/grants/home-energy-grants/solar-electricity-grant/) |
| SEAI grant | 900 € per kW up to 2 kW, i.e., 1,800 € for 2 kW | [SEAI, Solar Electricity Grant](https://www.seai.ie/grants/home-energy-grants/solar-electricity-grant/) |

The cell also draws the typical day of June with the two curves of Part 0.4, now for your family and your estimate.

### 4.3 The payback

The net cost is what the family pays after the grant. The payback is the net cost divided by the yearly saving. The cell draws the money balance of the family over 20 years. It starts at minus the net cost and climbs by the saving each year. The year in which the line crosses zero is the payback. Write it down. It is the second answer for your family. Then, edit the figures and run the cell again. Try a system of 4 kW, an export rate of 0.10 €, or a buying price of 0.45 €, and watch the crossing move. The cell keeps the saving constant over the years. A richer version would add the slow ageing of the panels, about 0.5% a year, and the changes of the prices.

## Part 5 — The reality

### 5.1 What the panels really produced

The cell loads the real production of your family in 2019 and draws it next to your estimate, month by month. Then, it prints the real energy, the real saving and the gap in percent. A month in which the records cover less than 90% of the half hours is marked as incomplete, and the yearly totals count only the complete months, for both the estimate and the reality. Write the gap of your family on the board, next to the gaps of the other teams.

Then, look for the causes of the gap. The usual ones are these.

- **The roof.** The formula assumes a roof tilted 30° towards the south. A roof that faces east or west receives 15 to 20% less radiation over a year. The study does not report the orientation of the four roofs.
- **The performance factor.** A value of 0.80 is typical for a new and clean system. Dirt, shade from a tree or a chimney, and a warm summer lower it.
- **The radiation itself.** The values of Open-Meteo come from a model with a grid of about 25 km. A coastal town with its own clouds can differ from the grid value.
- **The panels.** A system that stops for a week shows a month with less production. Compare the months one by one. A gap that appears in a single month points at the panels, and a gap that is the same in every month points at the roof or the factor.
- **The family.** The saving depends on the typical day of the family, which the notebook builds from records with gaps in some months.

### 5.2 The payback, again

The cell draws the money balance twice, with the saving of your projection and with the real saving. The distance between the two payback years is the cost of the gap for the family. An engineer who promises a payback of four years and delivers one of six has a problem. One who promises six and delivers five has a happy client.

## You have finished the practical

You started with the weather reconstructed from satellites and with the bills of a family. You trained a model that forecasts the solar radiation hour by hour, and you improved it by choosing its inputs and its size. You converted the forecast into electricity with a formula, into euro with the prices of the market, and into a payback with the cost of the system and the grant. Then, you compared your estimate with the measurement of a real roof and explained the gap. This is the complete path of an engineering estimate, from data to a decision, with a check against reality at the end.

## Something went wrong?

- If the setup cell cannot download the data files, run it again. If the error repeats, download the five files from "The data of the session", upload them in the Files panel of Colab (folder icon on the left), and run the setup cell once more.
- `RuntimeError: input.size(-1) must be equal to input_size`. The model expects a different number of inputs than the list `INPUTS` holds. Build the model again with `n_inputs=len(INPUTS)` after you change the list.
- `KeyError`. A name in `INPUTS` is not a column of the weather table. Copy the names from the table of Part 0.2.
- If a run takes more than a minute, reduce the epochs, the units or the window.
- If the loss shows `nan`, lower the learning rate.
- Part 4 needs at least one run in the log. If you restarted Colab, run the cells of Parts 1 to 3 again.

## Quick reference of the notebook

| Command | What it does |
|---|---|
| `RadiationModel(n_inputs, units, layers, dropout)` | builds the LSTM with a memory of `units` numbers |
| `run_experiment(name, model, inputs, window, epochs, learning_rate, batch_size, seed)` | trains the model, scores it on 2018 and adds a row to the log |
| `INPUTS = [...]` | the columns that the model reads at each hour |
| `show_log()` | prints the experiment log again |
| `energy_from_radiation(radiation, system_kw, tilt_factor, performance_factor)` | the formula of Part 0.6 |
| `savings(energy, consumption, buy_price, export_rate)` | the split between home and grid, and the saving in euro |
