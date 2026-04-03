import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_parquet('forecasts_filtered_rts3_constellation_v2.parquet')

# Filter to first 24hrs
df['TIME_HOURLY'] = pd.to_datetime(df['TIME_HOURLY'], utc=True)

# Midnight-to-midnight UTC
start = pd.Timestamp('2023-08-01 00:00:00', tz='UTC')
end   = pd.Timestamp('2023-08-02 00:00:00', tz='UTC')
df = df[(df['TIME_HOURLY'] >= start) & (df['TIME_HOURLY'] < end)]

grouped = df.groupby(['TIME_HOURLY', 'MODEL', 'ID_RESOURCE'])['FORECAST'].sum().reset_index()

resources = sorted(grouped['ID_RESOURCE'].unique())

# Distinct colors — avoid adjacent blues; use a hand-picked set if <=5 resources
PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b',
           '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff7f0e']
color_map = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(resources)}

fig, ax = plt.subplots(figsize=(14, 6))

for resource, rdf in grouped.groupby('ID_RESOURCE'):
    c = color_map[resource]
    # Individual model lines
    for model, mdf in rdf.groupby('MODEL'):
        mdf = mdf.sort_values('TIME_HOURLY')
        ax.plot(mdf['TIME_HOURLY'], mdf['FORECAST'],
                color=c, alpha=0.25, linewidth=0.7)
    # Mean across models at each timestep
    mean_df = rdf.groupby('TIME_HOURLY')['FORECAST'].mean().reset_index().sort_values('TIME_HOURLY')
    ax.plot(mean_df['TIME_HOURLY'], mean_df['FORECAST'],
            color=c, alpha=1.0, linewidth=2.0, label=resource)

ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax.set_xlabel('Time (UTC)')
ax.set_ylabel('Forecast')
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
plt.tight_layout()
plt.savefig('spaghetti_chart.png', dpi=150, bbox_inches='tight')
plt.savefig('spaghetti_chart.pdf', bbox_inches='tight')
print(f"Saved. Resources: {len(resources)}, Models: {grouped['MODEL'].nunique()}")