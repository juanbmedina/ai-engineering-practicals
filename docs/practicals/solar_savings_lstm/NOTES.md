# Practical 6 — Forecasting solar radiation with an LSTM to estimate the savings and payback of rooftop panels

Built 2 Nov 2026, revised 3 Nov 2026. Folder `docs/practicals/solar_savings_lstm/`.

## Files
- `index.md` — the practical page. The Colab badge still points to `COLAB_LINK_PENDING`: upload the STUDENT notebook to Drive and paste the `drive/<id>` link.
- `build_p6.py` — builds `MEEN41490_P6_Solar_Savings_STUDENT.ipynb` (27 cells) and, with `--teacher`, the TEACHER notebook (33 cells, worked attempts in Part 3).
- `MEEN41490_P6_Solar_Savings_TEACHER_executed.ipynb` — the teacher notebook run end to end on a 2-core CPU (158 s), house 1.
- `assets/` — Part 0 figures: `curve_of_a_panel.png`, `curve_of_a_house.png`, `radiation_and_panel_week.png` (from the real data, `figs_part0.py`), `panel_installation_tilt.png` and `lstm_architecture.png` (drawn, `figs_diagrams.py`), `mae_formula.png` (matplotlib mathtext). Two photos from Geograph Ireland, CC BY-SA 2.0, credited on the page: `rooftop_panels_bangor.jpg` ("Roof and solar panels, Bangor", Albert Bridge, geograph.ie/photo/3658336) and `rooftop_panels_country_house.jpg` ("All mod cons", Robert Ashby, geograph.ie/photo/5341132). No photo of the four Dingle houses exists under an open licence (the study has none, ESB Networks material is all rights reserved). `assets/data/` holds the five data files that the setup cell downloads from `DATA_URL`.
- Model design (3 Nov): the model is a visible `nn.Sequential` chain, as in P5, built from two helper blocks defined in the setup cell, `LSTMBlock(n_inputs, units)` (wraps `nn.LSTM`, returns the memory after each hour) and `LastHour()` (keeps the memory of the last hour), plus `nn.Linear`, `nn.ReLU`, `nn.Dropout`. The training loop squeezes the output. A "What you can change" table with every factor sits above the challenge cell and in Part 3 of the page.

## Data
- `dingle_weather_2010_2021.csv` — Open-Meteo archive API, ERA5, lat 52.141 lon -10.269 (Dingle), hourly UTC, 12 variables, tilt 30°, azimuth 0 (south), wind in m/s. CC BY 4.0, attribution "Open-Meteo / Copernicus Climate Change Service". 105,192 rows, 5.9 MB. The notebook uses 2010 to 2019.
- `house1.csv` to `house4.csv` — Avgoloupis et al. (2026), Scientific Data 13:834, Figshare 10.6084/m9.figshare.30294478, CC BY 4.0. Reduced to `timestamp, solar_kWh, import_kWh, export_kWh, elec_price` (the notebook drops `elec_price`, a tariff scenario of the study). 30-min, Jan 2019 to Feb 2022. Import and export are missing for about 70 days in Jan–Apr 2019 in every house. June–July 2021 production is zero in houses 2–4 (meter change), so the comparison year is **2019**, the cleanest year.
- Years: train 2010–2017, validation 2018, estimate and reality 2019.

## Results observed (seed 0, 2-core CPU)
- Baselines on 2018: same as this hour 36.3 W/m², same hour yesterday 51.6 W/m².
- First model (radiation only, 16 units, 5 epochs, lr 0.001): 22.9 W/m², 1,233 parameters, 7–20 s.
- 10 epochs at lr 0.003: about 20. + cloud_cover: 17.5–20.7 depending on the seed. + direct, diffuse, temperature, humidity, wind: 16.8–18.6. Two blocks of 32 + dropout + hidden layer: 16.3 in 42 s (13,969 parameters). 8 units: 18.4 (553 parameters). Hour of day and day of year add little. Target of the challenge: 20 W/m², under 60 s.
- Tilt factor 1.16. Performance factor 0.80. Estimate 2019 for 2.1 kW: about 1,960–1,990 kWh.
- Reality 2019 (complete months): house 1 +3 % gap (1,863 real), house 2 +18 % (Dec outage, 3 kWh), house 3 +43 % (10 complete months, low output all year), house 4 +11 %.
- Money (0.35 buy, 0.195 export, 2,100 €/kW, grant 1,800): net cost 2,610 €, projected saving 581–665 €/yr, payback 3.9–4.5 yr; real payback house 1 4.9 yr.

## Sources for the money figures (checked 2 Nov 2026)
- SEAI Solar Electricity Grant page: €900/kWp up to 2 kWp (€1,800 for 2 kWp), cap €2,400 with a battery, typical cost €1,700–2,500 per kWp. A Selectra page shows a lower 2026 table (cap €1,800); SEAI is the primary source.
- Clean Export Guarantee (Selectra table, updated 1 Nov 2026): SSE Airtricity and Electric Ireland 19.5 c/kWh, most others 18.5 c/kWh.
- Buying price 0.35 €/kWh is a round typical unit rate (Selectra uses 0.42 in one calculation).

## 4 Nov changes
- Every student works with house 1 (`HOUSE = 1`, fixed). The notebook downloads only the weather file and `house1.csv`; the other three house files stay on the site for reference.
- The setup cell has a fallback: if `DATA_URL` answers 404 (site not pushed yet) it builds the same files from the original sources, Open-Meteo archive API (`build_weather_from_open_meteo`) and Figshare (`build_house_from_figshare`, ndownloader ids 61423177/80/83/89). The 404 Juan saw on 4 Nov came from the folder not being committed and pushed yet.

- 4 Nov, second round: target of the challenge is now **MAE below 18 W/m²** (`TARGET_MAE = 18`, strict). Teacher path: 10 epochs lr 0.003 → 20.1, + cloud_cover → 18.3 (edge), seven inputs → 16.8 / 17.5 (seed 1), big chain → 16.3 in 35 s, 8 units → 18.4 (above). `plot_week` has fixed axes (7 days, 0–1000 W/m², one tick per day). The LSTM figure is now a three-panel pipeline (window → one LSTM with a memory loop → forecast of hour 25), `fig_lstm2.py`. The factor table in the notebook has three columns so it fits the cell. The MAE sentence on the page no longer uses the | characters (kramdown read them as a table). Section 0.1 names the institutions once and leaves the authors to the citation.

## 5 Nov changes (text revision after David's feedback)
- David's feedback: the text read as generated. He asked for a direct, academic register, without hooks or abstractions, and pointed out titles that start with Wh, titles of the form "X, Y", and over-justification of obvious points.
- `index.md` and every Markdown cell of `build_p6.py` rewritten. Titles are now descriptive noun phrases (e.g., "0.4 Daily and seasonal production profile of a solar panel", "Common errors and solutions"), with no Wh/How openings, no commas and no questions. The role-play of "the engineer of a family" is replaced by a direct statement of the two quantities to estimate. "Family" is now "household", "simple guesses" are "baselines", "the reality" is "the measured production". New terms defined at first use: reanalysis, self-consumption, typical day, baseline, validation year, overfitting.
- Prose mean sentence length went from about 15 to about 20 words (page, troubleshooting excluded). No while/rather/semicolons in the prose.
- Corrections made on the way: the notebook text claimed that the log table shows the two baselines (it does not), and the weather row of 12:00 is the hour that ends at 12:00 (Open-Meteo averages the preceding hour).
- Visible code strings updated to match (cell titles, print labels, plot labels and titles). Variable names unchanged. STUDENT and TEACHER notebooks regenerated (27 and 33 cells). The `_executed` notebook still has the old text and outputs.
- To do: upload the new STUDENT notebook to Drive (replace the file behind the current Colab link, or paste the new link in `index.md`).

## To do
- Upload STUDENT to Drive, paste the Colab link in `index.md`, commit and push (the setup cell needs `assets/data/` live on github.io).
- Optional: re-execute the teacher notebook in Colab to refresh `_executed`.
