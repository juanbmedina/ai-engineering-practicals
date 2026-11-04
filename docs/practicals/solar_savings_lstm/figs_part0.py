import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
w = pd.read_csv('data/dingle_weather_2010_2021.csv', parse_dates=['time']).set_index('time')
d = pd.read_csv('data/house1.csv', parse_dates=['timestamp']).set_index('timestamp')
# ---- Figure 1: curve of a solar panel: clear day, cloudy day, and monthly totals (house 1, 2019)
g = w['shortwave_radiation']
june = d['solar_kWh']['2019-06-01':'2019-06-30']
daily = june.resample('D').sum()
clear = daily.idxmax().strftime('%Y-%m-%d'); cloudy = daily[daily > 0.3].idxmin().strftime('%Y-%m-%d')
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), gridspec_kw={'width_ratios': [1, 1, 1.3]})
for ax, day, title in zip(axes[:2], [clear, cloudy], ['A clear day', 'A cloudy day']):
    s = d['solar_kWh'][day] * 2   # kWh per half hour -> kW
    ax.plot(s.index.hour + s.index.minute / 60, s.values, color='#f4a261', linewidth=2)
    ax.fill_between(s.index.hour + s.index.minute / 60, s.values, color='#f4a261', alpha=0.3)
    ax.set_xlim(0, 24); ax.set_ylim(0, 2.3); ax.set_xticks([0, 6, 12, 18, 24]); ax.set_xlabel('hour of the day'); ax.set_ylabel('kW')
    ax.set_title(f'{title}, {pd.Timestamp(day).strftime("%d %B %Y")}, 2.1 kW roof'); ax.grid(alpha=0.3)
m = d['solar_kWh']['2019'].resample('ME').sum()
axes[2].bar(range(1, 13), m.values, color='#f4a261')
axes[2].set_xticks(range(1, 13)); axes[2].set_xticklabels([t.strftime('%b') for t in m.index], fontsize=8)
axes[2].set_ylabel('kWh per month'); axes[2].set_title('The same roof over the year 2019'); axes[2].grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig('p6/assets/curve_of_a_panel.png', dpi=130); plt.close()
print('clear', clear, 'cloudy', cloudy, 'year total', m.sum().round(0))
# ---- Figure 2: curve of a house: January and June profiles, with the panel profile of June
cons = (d['import_kWh'] + d['solar_kWh'] - d['export_kWh']).clip(lower=0)['2019'].resample('h').sum(min_count=2)
pv = d['solar_kWh']['2019'].resample('h').sum(min_count=2)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for month, colour, label in [(12, '#264653', 'December'), (6, '#2a9d8f', 'June')]:
    c = cons[cons.index.month == month]; prof = c.groupby(c.index.hour).mean()
    axes[0].plot(prof.index, prof.values, color=colour, linewidth=2, label=f'consumption, {label}')
axes[0].set_xlim(0, 23); axes[0].set_ylim(0); axes[0].set_xlabel('hour of the day'); axes[0].set_ylabel('kWh per hour')
axes[0].set_title('A typical day of the family (house 1, 2019)'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
c = cons[cons.index.month == 6]; prof = c.groupby(c.index.hour).mean()
p = pv[pv.index.month == 6]; pprof = p.groupby(p.index.hour).mean()
axes[1].plot(prof.index, prof.values, color='#2a9d8f', linewidth=2, label='consumption of the family')
axes[1].plot(pprof.index, pprof.values, color='#f4a261', linewidth=2, label='production of the panels')
axes[1].fill_between(prof.index, np.minimum(prof.values, pprof.reindex(prof.index).values), color='#f4a261', alpha=0.35, label='used at home')
axes[1].set_xlim(0, 23); axes[1].set_ylim(0); axes[1].set_xlabel('hour of the day'); axes[1].set_ylabel('kWh per hour')
axes[1].set_title('June: the panels peak at midday, the family in the evening'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('p6/assets/curve_of_a_house.png', dpi=130); plt.close()
print('June: used at home share', (np.minimum(prof.values, pprof.reindex(prof.index).values).sum() / pprof.sum()).round(2))
# ---- Figure 3: radiation from satellite vs the real panel, one week of June 2019
fig, ax = plt.subplots(figsize=(11, 3.2))
wk = slice('2019-06-17', '2019-06-23')
ax.plot(g[wk].index, g[wk].values, color='#444', label='solar radiation from Open-Meteo, W/m²')
ax2 = ax.twinx(); s = (d['solar_kWh'][wk] * 2)
ax2.plot(s.index, s.values, color='#f4a261', label='production of the panels of house 1, kW')
ax.set_ylabel('W/m²'); ax2.set_ylabel('kW'); ax.set_title('One week of June 2019: the radiation reconstructed from satellites and the real production of a roof')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('p6/assets/radiation_and_panel_week.png', dpi=130); plt.close()
# sample rows for the page
print(d['2019-06-21 11:00':'2019-06-21 12:30'].round(3).to_string())
print(w[['shortwave_radiation','direct_radiation','diffuse_radiation','global_tilted_irradiance','cloud_cover','temperature_2m','wind_speed_10m']]['2019-06-21 10:00':'2019-06-21 13:00'].round(1).to_string())
