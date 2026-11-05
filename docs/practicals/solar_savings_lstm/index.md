---
layout: default
title: "Practical 6 — Forecasting solar radiation with an LSTM"
---

# Practical 6 — Forecasting solar radiation with an LSTM

## Organisation of the session

In this practical, you estimate two quantities for a household in Dingle, Co. Kerry, that took part in a study on residential solar energy. The first quantity is the yearly electricity production of a 2.1 kW rooftop solar system. The second quantity is the payback time, which is the number of years that the savings on the electricity bill take to cover the net cost of the system. Both estimates rely on a forecast of the hourly solar radiation, which you build with a neural network in a Colab notebook.

| Part | Content |
|---|---|
| 0 — This page | Origin of the data, production profile of a solar panel, consumption profile of a household, and conversion of radiation into electricity |
| 1 to 3 — Notebook | Training of a model that forecasts the solar radiation, and a challenge to improve it |
| 4 — Notebook | Conversion of the forecast into electricity, savings and payback time |
| 5 — Notebook | Comparison of the estimate with the measured production of the household |

In Practical 5, the model received an image and returned a class, either overalls or no overalls. In this practical, the model receives the weather of the previous hours and returns a number, which is the forecast of the solar radiation for the next hour. This model is a Long Short-Term Memory network (LSTM), a type of neural network that keeps an internal memory as it reads a sequence of values. The notebook follows the working pattern of Practical 5, with the model written as a chain of layers and every training run recorded in an experiment log. In the challenge, you edit the inputs and the layers of the model directly in the code.

## Part 0 — Background for the notebook

### 0.1 Household data from the Dingle pilot study

Between 2019 and 2022, four households on the Dingle Peninsula took part in a pilot project of ESB Networks, the operator of the electricity distribution network in Ireland. Each house received a 2.1 kW rooftop solar system, a home battery, an electric car with its charger, and a heat pump. Smart meters recorded the electricity that each house produced, imported from the grid, and exported to the grid every 30 minutes for three years. Researchers from the National Technical University of Athens and EPRI Europe (Dublin) published these records in 2026 under an open licence.

> Avgoloupis, D., Michalakopoulos, V., Sarmas, E., Marinakis, V., Couto, M., Coccia, A. and Jimenez, M. (2026). *Residential Energy Dataset with Electric Vehicles, Photovoltaic Generation and Tariff Variability in Ireland.* Scientific Data, 13, 834. [https://doi.org/10.1038/s41597-026-07186-3](https://doi.org/10.1038/s41597-026-07186-3). Data on Figshare, [https://doi.org/10.6084/m9.figshare.30294478](https://doi.org/10.6084/m9.figshare.30294478), licence CC BY 4.0.

All students work with house 1 of the study, whose consumption records play the role of the electricity bills that a household provides to an engineer before an installation. The measured production of its panels is the reference for the comparison of Part 5.

### 0.2 Solar radiation from the ERA5 reanalysis

The solar radiation values of this practical come from a **reanalysis**. A reanalysis is a reconstruction of past weather that combines satellite images, measurements of ground stations, and a physical model of the atmosphere. The European Centre for Medium-Range Weather Forecasts (ECMWF) produces the ERA5 reanalysis, which covers the whole planet hour by hour from 1940 to the present. The Copernicus Climate Change Service of the European Union distributes ERA5 free of charge. The notebook obtains the ERA5 values through **Open-Meteo**, a free web service that returns the hourly weather for any pair of coordinates. The weather file covers Dingle, at latitude 52.14° N and longitude 10.27° W, from 2010 to 2021, with the columns of the table below.

| Column | Description | Unit |
|---|---|---|
| `shortwave_radiation` | **global horizontal radiation**, the solar power that reaches one square metre of flat ground, averaged over the hour | W/m² |
| `direct_radiation` | part of the radiation that comes straight from the sun | W/m² |
| `diffuse_radiation` | part of the radiation scattered by the sky and the clouds | W/m² |
| `direct_normal_irradiance` | direct radiation on a surface that faces the sun | W/m² |
| `global_tilted_irradiance` | global radiation on a roof that faces south with a tilt of 30° | W/m² |
| `cloud_cover`, `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high` | share of the sky covered by clouds, in total and by height | % |
| `temperature_2m` | air temperature two metres above the ground | °C |
| `relative_humidity_2m` | relative humidity | % |
| `wind_speed_10m` | wind speed ten metres above the ground | m/s |

Radiation is a power per unit area, expressed in watts per square metre (W/m²). A constant radiation of 1000 W/m² during one hour delivers 1 kWh of energy to each square metre of surface. In Dingle, the radiation at noon reaches 700 to 900 W/m² on a clear summer day and stays below 100 W/m² on a cloudy winter day. These values come from a model and not from a sensor on the roof. Therefore, part of the difference between the estimate and the measured production, analysed in Part 5, originates in the radiation data.

The figure below compares the Open-Meteo radiation, in grey, with the measured production of the panels of house 1, in orange, over one week of June 2019. Both curves follow the same daily cycle, and the cloudy days appear as drops in both.

![One week of June 2019 with two curves: the solar radiation reconstructed from satellites, in W per square metre, and the real production of the panels of house 1, in kW. The curves rise and fall together each day, with dents on the cloudy days]({{ '/practicals/solar_savings_lstm/assets/radiation_and_panel_week.png' | relative_url }})

### 0.3 Tilt and orientation of a rooftop installation

A domestic solar system consists of photovoltaic panels mounted on a frame that follows the slope of the roof. With a typical panel of about 400 W under a radiation of 1000 W/m², the 2.1 kW systems of the Dingle study correspond to five or six panels. An inverter converts the direct current of the panels into the alternating current that the house uses. A bidirectional meter records the electricity that the house imports from the grid and exports to it. The two photographs below show domestic installations in Bangor, Co. Down, and near Moneydig, Co. Londonderry.

![A semi-detached house in Bangor, Co. Down, with twelve solar panels in two rows on the slope of the roof, next to the chimney]({{ '/practicals/solar_savings_lstm/assets/rooftop_panels_bangor.jpg' | relative_url }})

![A white bungalow on a country road near Moneydig, Co. Londonderry, with a row of solar panels on the slope of its roof that faces the road]({{ '/practicals/solar_savings_lstm/assets/rooftop_panels_country_house.jpg' | relative_url }})

*Photos: "Roof and solar panels, Bangor" by Albert Bridge and "All mod cons" by Robert Ashby, from [Geograph Ireland](https://www.geograph.ie/), licence CC BY-SA 2.0.*

Two geometric parameters of the roof determine the radiation that the panels receive. The **tilt** is the angle between the roof and the horizontal plane. The **orientation** is the compass direction that the slope faces. In Ireland, the sun at noon stands about 15° above the southern horizon in December and about 60° in June. A panel receives the most radiation when its surface is perpendicular to the direction of the sun. Therefore, a roof that faces south with a tilt of 30° to 35° collects the most energy over a year. The figure below shows this geometry and the yearly radiation that a panel in Dingle receives for different tilts and orientations.

![Left, a house seen from the side with panels on the slope that faces south, the tilt angle of 30 degrees marked, and the sun high in June and low in December. Right, a chart of the yearly radiation received by a panel in Dingle against its tilt, for a roof that faces south, east or west, and north]({{ '/practicals/solar_savings_lstm/assets/panel_installation_tilt.png' | relative_url }})

Over a year, a roof that faces south with a tilt of 30° receives about 16% more radiation than flat ground. In contrast, a roof that faces east or west receives about the same radiation as flat ground, and a roof that faces north receives about 25% less. The study does not report the orientation of the four roofs, so the notebook assumes a south-facing roof with a tilt of 30°. Part 5 discusses the error that this assumption can introduce.

### 0.4 Daily and seasonal production profile of a solar panel

The production of a solar panel follows the daily path of the sun, with a maximum around solar noon and no output between sunset and sunrise. On a clear day, the daily production follows a smooth bell-shaped curve. On a cloudy day, the clouds cause irregular drops in this curve, and the peak can be as low as a quarter of the clear-day value. Over the year, the curve is higher and wider in June than in December, following the length of the day and the height of the sun.

![Three panels: the production of a 2.1 kW roof on a clear day of June 2019, a smooth bell that peaks near 1.9 kW. The same roof on a cloudy day, a jagged curve under 0.5 kW. The monthly production of the roof over 2019, from under 40 kWh in January and December to about 300 kWh in June]({{ '/practicals/solar_savings_lstm/assets/curve_of_a_panel.png' | relative_url }})

The figure uses the measured production of house 1 on two days of June 2019 and over the whole year. On 18 June 2019, a clear day, the 2.1 kW system reached a peak of about 1.9 kW. In contrast, on 23 June, a cloudy day, the output stayed below 0.5 kW. In 2019, the system produced 1,881 kWh, with 297 kWh in June and 17 kWh in January. This total corresponds to 896 kWh per kW installed, which lies within the usual range of 850 to 950 kWh per kW for Ireland.

### 0.5 Daily consumption profile of a household

The electricity consumption of a household is low at night, rises in the morning, decreases around midday when the occupants are often away, and reaches its maximum in the evening. Daily consumption is also higher in winter than in summer, as the left panel of the figure shows for the typical day of house 1 in December and in June. The **typical day** is the average consumption at each hour of the day over one month.

![Two panels: the typical daily consumption of house 1 in December and in June, with a morning bump and an evening peak. The June curve of the household next to the June curve of the panels, with the overlap of the two shaded as the part used at home]({{ '/practicals/solar_savings_lstm/assets/curve_of_a_house.png' | relative_url }})

The right panel overlays the June consumption profile and the June production profile of the panels. The production peaks at midday, whereas the consumption peaks in the morning and in the evening. The shaded area represents the **self-consumption** of the household, defined as the electricity that it uses at the moment the panels produce it. The rest of the production is exported to the grid, and each of the two parts has a different value for the household.

- Self-consumed electricity replaces electricity that the household would otherwise buy, so its value is the **buying price**, about 0.35 € per kWh in November 2026.
- Exported electricity is paid by the supplier at the **export rate**, about 0.195 € per kWh in November 2026.

Consequently, two households with identical systems obtain different savings when their consumption profiles differ. Part 4 computes the saving of house 1 with this split between self-consumption and export.

### 0.6 Structure of the household and weather files

The file of each household contains one row per half hour from January 2019 to February 2022. The notebook uses the four columns described in the table below.

| Column | Description | Unit |
|---|---|---|
| `timestamp` | start of the half hour, in Coordinated Universal Time (UTC) | date and time |
| `solar_kWh` | electricity produced by the panels during the half hour | kWh |
| `import_kWh` | electricity imported from the grid during the half hour | kWh |
| `export_kWh` | electricity exported to the grid during the half hour | kWh |

The table below shows four rows of house 1 around midday on 21 June 2019.

| `timestamp` | `solar_kWh` | `import_kWh` | `export_kWh` |
|---|---|---|---|
| 2019-06-21 11:00 | 0.880 | 0.066 | 0.257 |
| 2019-06-21 11:30 | 0.924 | 0.046 | 0.528 |
| 2019-06-21 12:00 | 0.927 | 0.060 | 0.590 |
| 2019-06-21 12:30 | 0.886 | 0.000 | 0.694 |

In the last row, the panels produced 0.886 kWh, the household imported nothing, and 0.694 kWh went to the grid. Therefore, the household consumed 0.886 − 0.694 = 0.192 kWh of its own production in that half hour. In general, the consumption of each half hour follows from the three measured columns.

> consumption = `import_kWh` + `solar_kWh` − `export_kWh`

The notebook applies this equation to build the typical day of the household for each month. The original dataset also contains battery and heat pump records for the last months of the study and a tariff scenario defined by the researchers, which the notebook does not use.

The weather file contains one row per hour, with `shortwave_radiation` as the variable that the model forecasts and the other columns of Section 0.2 as candidate inputs. The table below shows the hours around midday of 21 June 2019.

| `time` | `shortwave_radiation` | `cloud_cover` |
|---|---|---|
| 2019-06-21 10:00 | 601 | 100 |
| 2019-06-21 11:00 | 646 | 99 |
| 2019-06-21 12:00 | 735 | 81 |
| 2019-06-21 13:00 | 735 | 47 |

In the last row, the radiation on flat ground averaged 735 W/m² over the hour that ends at 13:00, and clouds covered 47% of the sky. Because the household file uses half hours and the weather file uses hours, the notebook adds the two half-hour values of each hour to obtain hourly values.

### 0.7 Conversion of solar radiation into electrical energy

The notebook uses the following formula to convert the hourly radiation into the energy that the panels produce.

> energy in one hour (kWh) = radiation (W/m²) / 1000 × tilt factor × system power (kW) × performance factor

- **radiation / 1000** is the energy, in kWh, that reaches one square metre of flat ground during the hour.
- **The tilt factor** is the ratio between the radiation on the tilted roof and the radiation on flat ground. For a roof in Dingle that faces south with a tilt of 30°, the notebook computes this factor from the weather file and obtains about 1.16 over a year.
- **The system power** is the power that the panels deliver under a radiation of 1000 W/m², and its value for house 1 is 2.1 kW.
- **The performance factor** accounts for the losses due to dirt, panel temperature, cables, and the inverter. A new system typically delivers about 80% of its ideal output, so the notebook uses a factor of 0.80.

For example, the hour that ends at 12:00 on 21 June 2019 gives 735 / 1000 × 1.16 × 2.1 × 0.80 = 1.43 kWh. The panels of house 1 produced 0.880 + 0.924 = 1.80 kWh in the two corresponding half hours, so the formula underestimates the production of that hour. A cloudy winter hour with 50 W/m² gives 50 / 1000 × 1.16 × 2.1 × 0.80 = 0.10 kWh. Design tools for solar installations use more detailed versions of this formula. However, over a whole year, the simple formula stays within a few percent of the measured production when the roof faces south and the system operates without faults. Part 5 tests this statement for house 1.

### 0.8 Time-series forecasting with an LSTM

A **time series** is a sequence of measurements ordered in time, e.g., the hourly radiation in Dingle. A **forecast** estimates the next value of a time series from its past values. The model of this practical reads a **window** of past hours, e.g., the radiation and the cloud cover of the last 24 hours, and returns the radiation of the next hour. To forecast the following hour, the window moves one hour forward and the model repeats the same computation. The figure below shows how the LSTM processes a window of 24 hours.

![Three panels. Left, the window: a bar chart of the radiation of the last 24 hours, read from the oldest to the newest. Middle, one LSTM box that reads the inputs of one hour and updates a memory, with a loop arrow that passes the memory to the next hour. Right, the same 24 hours with the forecast of hour 25 as a green dot next to the truth]({{ '/practicals/solar_savings_lstm/assets/lstm_architecture.png' | relative_url }})

A single LSTM processes the 24 hours of the window one at a time, from the oldest hour to the most recent one. At each hour, the LSTM receives the inputs of that hour and updates a **memory**. The memory is a short list of numbers, whose length is the number of **units**. At each update, the LSTM decides which information to add to the memory and which information to discard. The LSTM applies the same weights at every hour, so the number of parameters does not depend on the length of the window. After the last hour, the memory summarises the whole window, and a linear layer, as in Practical 5, combines its values into the forecast. In the notebook, the model is a chain of three layers.

- `LSTMBlock` reads the window and returns the memory after each hour.
- `LastHour` keeps only the memory after the last hour.
- `nn.Linear` converts this memory into the forecast.

The notebook evaluates the forecasts with the **mean absolute error (MAE)**. For a year of N hours, the MAE is the average of the absolute differences between the forecast and the actual value at each hour.

![The formula of the mean absolute error: MAE equals one over N times the sum, over the N hours of the year, of the absolute difference between the forecast and the truth at each hour]({{ '/practicals/solar_savings_lstm/assets/mae_formula.png' | relative_url }})

The MAE has the same unit as the radiation, W/m², and a lower MAE indicates a better forecast. Two simple forecasts that need no model, called **baselines**, set the minimum performance that a useful model must exceed. The first baseline assumes that the radiation of the next hour equals the radiation of the current hour, and its MAE in Dingle is about 36 W/m². The second baseline assumes that it equals the radiation at the same hour of the previous day, and its MAE is about 52 W/m².

## Content of the notebook

The notebook is divided into the five parts listed below.

| Part | Content |
|---|---|
| 1 — Solar radiation data for Dingle | Loading of the weather data, plots of one year, one week of June and one week of December, and chronological split of the years |
| 2 — Baseline forecasts and a first LSTM model | Evaluation of the two baselines, description of the training loop, and training of a first LSTM |
| 3 — Challenge to improve the radiation forecast | Modification of the inputs and the model to reach the target MAE |
| 4 — Monetary projection | Conversion of the forecast into electricity, savings and payback time |
| 5 — Comparison with the measured production | Comparison of the estimate with the measured production of house 1 |

> **Before you begin.** If you have not used Colab before, read [Getting Started with Google Colab]({{ '/guides/getting-started-colab/' | relative_url }}) first. The notebook runs on the CPU, so a connection without GPU is sufficient. The setup cell downloads the two data files automatically.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uKsBb9oCn3nxgnGgFXhAA2fR6M20nP-s)

### Data files

The setup cell downloads the weather file and the file of house 1 into Colab automatically. If the course website is unavailable, the cell retrieves the same data from Open-Meteo and Figshare. The links below serve only to inspect the data.

- [`dingle_weather_2010_2021.csv`]({{ '/practicals/solar_savings_lstm/assets/data/dingle_weather_2010_2021.csv' | relative_url }}){:download=""}, hourly weather of Dingle from Open-Meteo (ERA5, Copernicus Climate Change Service), 2010 to 2021, 5.9 MB.
- [`house1.csv`]({{ '/practicals/solar_savings_lstm/assets/data/house1.csv' | relative_url }}){:download=""}, [`house2.csv`]({{ '/practicals/solar_savings_lstm/assets/data/house2.csv' | relative_url }}){:download=""}, [`house3.csv`]({{ '/practicals/solar_savings_lstm/assets/data/house3.csv' | relative_url }}){:download=""}, [`house4.csv`]({{ '/practicals/solar_savings_lstm/assets/data/house4.csv' | relative_url }}){:download=""}, half-hourly records of the four households of the study, 2019 to 2022, 2.3 MB each, from Avgoloupis et al. (2026), CC BY 4.0. The notebook uses only `house1.csv`.

## Part 1 — Solar radiation data for Dingle

### 1.1 Loading and visualisation of the weather data

The cell loads the weather file, prints its columns and four rows of 21 June 2019, and draws three figures. The first figure shows the total yearly radiation on flat ground, which ranges from about 1,000 to 1,100 kWh/m² in Dingle, with differences below 10% between years. The other two figures show the global, direct and diffuse radiation over one week of June and one week of December. In these figures, observe the height of the daily peaks, the times of sunrise and sunset, and the effect of the clouds on the direct radiation.

### 1.2 Chronological split of the data

The model trains on the years 2010 to 2017, and the challenge evaluates it on 2018, which is the **validation year**. The year 2019 is kept aside for the estimate of Part 4 and the comparison of Part 5. This chronological split evaluates the model on hours later than those of the training, as in a real forecast. The cell also computes the **scale** of each column, defined as its maximum value over the training years. Each input is divided by its scale before it enters the model, so that all inputs lie between 0 and 1.

## Part 2 — Baseline forecasts and a first LSTM model

### 2.1 Baseline forecasts

The cell computes the MAE of the two baselines of Section 0.8 on the validation year, which gives the reference values for every model of the challenge.

### 2.2 Training loop and experiment log

The cell defines the two functions that the challenge uses, `train_model` and `run_experiment`. The function `train_model` runs the gradient steps, as in Practical 5, with the squared difference between the forecast and the actual radiation as the loss. This loss is the usual choice when the output of a model is a number instead of a class. The function `run_experiment` cuts the hours into windows with the selected inputs and trains a new model from random initial parameters. Then, it computes the MAE on 2018, plots the loss and one week of June 2018 with the actual and the forecast radiation, and adds one row to the **experiment log**.

### 2.3 A first LSTM model

As in Practical 5, the model is a chain of layers defined inside `nn.Sequential`. In the first model, `LSTMBlock(1, 16)` reads the window one hour at a time with a memory of 16 units, `LastHour()` keeps the memory after the last hour, and `nn.Linear(16, 1)` converts it into the forecast. This model reads only the radiation of the last 24 hours and trains for 5 epochs with a learning rate of 0.001. It has 1,233 parameters, trains in about ten seconds, and reaches an MAE of about 23 W/m², below both baselines. The loss is still decreasing at the end of the training, which indicates that more epochs can improve the model.

## Part 3 — Challenge to improve the radiation forecast

The goal of the challenge is a model with a **validation MAE below 18 W/m²** and a training time of **at most 60 seconds**. Among the models that reach this target, the lowest MAE wins, and the smaller number of parameters breaks a tie. The experiment log marks with "yes" in its last column every run that meets both conditions. Because the MAE varies by one or two W/m² between random seeds, run a good configuration again with `seed=1` to confirm that it reaches the target with both seeds.

### Adjustable factors of the model and the training

The table below lists the factors that you can modify, with their effect and the expected result of a change, and the notebook repeats it above the challenge cell.

| Factor | Location | Effect | Expected result |
|---|---|---|---|
| `INPUTS` | list at the top of the cell | columns that the model reads at each hour of the window, from the table of Section 0.2 plus `hour_of_day` and `day_of_year` | additional inputs give the model more information, e.g., the cloud cover indicates whether the sky is clearing |
| `window` | argument of `run_experiment` | number of past hours that the model reads before each forecast | 12 hours trains fast, 24 hours covers one day, and 48 hours covers two days at twice the training time |
| units of `LSTMBlock(n_inputs, units)` | the chain | size of the memory | 8 units is small and fast, 16 is the default, and 32 or 64 retain more information at a longer training time |
| second `LSTMBlock(units, units2)` | the chain, after the first block | second memory that reads the memories of the first block | slightly higher accuracy at twice the training time, and its first argument must equal the units of the previous block |
| `nn.Linear(units, hidden)` + `nn.ReLU()` | the chain, after `LastHour()` | hidden layer between the memory and the forecast, as in Practical 5 | useful when the memory is large, and the next `nn.Linear` must start with `hidden` |
| `nn.Dropout(0.1)` | the chain, between layers | sets 10% of the values to zero at random during training | reduces the dependence of the model on individual values, and may increase the training MAE slightly |
| `epochs` | argument of `run_experiment` | number of passes over all the training windows | more epochs reduce the loss until it flattens, at about one second per epoch |
| `learning_rate` | argument of `run_experiment` | size of each gradient step | 0.001 is slow and stable, 0.003 is a good default, and 0.01 is fast but can oscillate |
| `batch_size` | argument of `run_experiment` | number of windows processed before each step | 256 is the default, and 64 gives more steps per epoch at a longer training time |
| `seed` | argument of `run_experiment` | random initialisation of the parameters | changes the MAE by one or two W/m², so a good configuration needs a check with two seeds |

**Recommended procedure.** Change one factor at a time, give each run a name that describes the change, and inspect the loss and the weekly plot before the next change. A loss that is still decreasing at the end of the training indicates that more epochs can help. A loss that oscillates indicates that the learning rate is too high. A forecast that lags one hour behind the actual radiation indicates that the model copies the last hour, and the cloud inputs can help it to anticipate changes. As in Practical 5, the first argument of each layer must equal the number of values that the previous layer returns. A model with 16 units trained for 10 epochs takes about 15 seconds, which leaves time for many runs.

## Part 4 — Monetary projection

### 4.1 Estimated electricity production in 2019

The cell selects the model with the lowest validation MAE in the log and forecasts the radiation for every hour of 2019. Then, it applies the formula of Section 0.7 with a system power of 2.1 kW, the tilt factor computed from the weather file, and a performance factor of 0.80. The cell prints the estimated production per month and the yearly total. This yearly total is the first of the two quantities that you estimate in this practical.

### 4.2 Estimated yearly saving

The cell builds the typical day of house 1 for each month from its records, with the consumption equation of Section 0.6. Then, it splits the estimated production of each hour into self-consumption and export, and values each part at its price. The prices correspond to November 2026, and you can modify them in the cell. The cost of the system and the grant come from the Sustainable Energy Authority of Ireland (SEAI).

| Parameter | Value | Source |
|---|---|---|
| Buying price of electricity | 0.35 € per kWh | typical unit rate of the Irish suppliers, November 2026 |
| Export rate | 0.195 € per kWh | Clean Export Guarantee of SSE Airtricity and Electric Ireland, November 2026, [Selectra](https://selectra.ie/energy/guides/prices/best-microgeneration-rates-ireland) |
| Cost of a system | 1,700 to 2,500 € per kW installed | [SEAI, Solar Electricity Grant](https://www.seai.ie/grants/home-energy-grants/solar-electricity-grant/) |
| SEAI grant | 900 € per kW up to 2 kW (1,800 € for 2 kW) | [SEAI, Solar Electricity Grant](https://www.seai.ie/grants/home-energy-grants/solar-electricity-grant/) |

The cell also plots the typical June day of house 1 against the estimated production, in the same format as the right panel of Section 0.5.

### 4.3 Payback time

The **net cost** is the cost of the system minus the SEAI grant. The **payback time** is the net cost divided by the yearly saving. The cell plots the cumulative balance of the household over 20 years, which starts at minus the net cost and increases by the yearly saving every year. The year in which this balance crosses zero is the payback time. This value is the second quantity that you estimate in this practical. To test the sensitivity of the result, edit the parameters and run the cell again, e.g., with a system of 4 kW, an export rate of 0.10 € per kWh, or a buying price of 0.45 € per kWh. The calculation assumes a constant yearly saving, whereas a more complete calculation would include the degradation of the panels, about 0.5% per year, and the evolution of electricity prices.

## Part 5 — Comparison with the measured production

### 5.1 Estimated and measured production in 2019

The cell loads the measured production of house 1 in 2019 and plots it next to the estimate, month by month. Then, it prints the measured energy, the corresponding saving, and the gap between the estimate and the measurement in percent. A month in which the records cover less than 90% of the half hours is marked as incomplete. The yearly totals of both the estimate and the measurement include only the complete months. Each student writes the gap on the board. Since all students work with the same house, the differences between the gaps come from the models.

The usual causes of the gap are the following.

- **Orientation and tilt of the roof.** The formula assumes a roof that faces south with a tilt of 30°. A roof that faces east or west receives 15 to 20% less radiation over a year than a roof that faces south. The study does not report the orientation of the four roofs.
- **Performance factor.** The value of 0.80 is typical for a new and clean system. Dirt, shading from a tree or a chimney, and high summer temperatures reduce it.
- **Radiation data.** The Open-Meteo values come from a model with a grid of about 25 km. On a coastal site with local cloud patterns, the radiation can differ from the value of the grid cell.
- **Faults of the system.** A system that stops for a week produces a month with a low total. A gap concentrated in a single month points to a fault, whereas a similar gap in every month points to the roof geometry or the performance factor.
- **Consumption profile.** The saving depends on the typical day of the household, which the notebook builds from records with gaps in some months.

### 5.2 Estimated and measured payback time

The cell plots the cumulative balance twice, once with the estimated saving and once with the measured saving. The difference between the two payback times quantifies the financial consequence of the estimation error for the household.

## Summary of the practical

In this practical, you trained an LSTM that forecasts the hourly solar radiation in Dingle from reanalysis data. You improved its accuracy through the choice of inputs, the size of the memory, and the training settings. Then, you converted the forecast into electricity with a conversion formula, into a yearly saving with current market prices, and into a payback time with the cost of the system and the grant. Finally, you compared the estimate with the measured production of house 1 and identified the sources of the gap.

## Common errors and solutions

- If the setup cell cannot download the data files, run it again. If the error persists, download the two files from [Data files](#data-files), upload them in the Files panel of Colab (folder icon on the left), and run the setup cell once more.
- `RuntimeError: input.size(-1) must be equal to input_size`. The first `LSTMBlock` expects a different number of inputs than the list `INPUTS` holds. Keep `LSTMBlock(len(INPUTS), ...)` as the first layer and run the cell again after you change the list.
- `RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x32 and 16x1)`. A `nn.Linear` layer expects 16 values but receives 32. Set its first argument to the units of the previous layer.
- `KeyError`. A name in `INPUTS` is not a column of the weather table. Copy the names from the table of Section 0.2.
- If a run takes more than a minute, reduce the epochs, the units or the window.
- If the loss shows `nan`, lower the learning rate.
- Part 4 needs at least one run in the log. If you restarted Colab, run the cells of Parts 1 to 3 again.

## Quick reference of the notebook functions

| Command | Description |
|---|---|
| `LSTMBlock(n_inputs, units)`, `LastHour()` | the two blocks of the chain specific to this practical, used together with `nn.Linear`, `nn.ReLU` and `nn.Dropout` from Practical 5 |
| `run_experiment(name, model, inputs, window, epochs, learning_rate, batch_size, seed)` | trains the model, computes its MAE on 2018 and adds a row to the log |
| `INPUTS = [...]` | the columns that the model reads at each hour |
| `show_log()` | prints the experiment log again |
| `energy_from_radiation(radiation, system_kw, tilt_factor, performance_factor)` | the formula of Section 0.7 |
| `savings(energy, consumption, buy_price, export_rate)` | splits the production into self-consumption and export, and returns the saving in euro |
