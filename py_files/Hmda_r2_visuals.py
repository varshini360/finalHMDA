import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np



df = pd.read_csv("/Users/varshininarayanan/Downloads/county_disparities_full.csv")


FIPS_COL = "county_code"
GAP_COL = "approval_disparity_gap"
STATE_COL = "state_code"


df["county_fips"] = (
    df[FIPS_COL]
    .astype(int)
    .astype(str)
    .str.zfill(5)
)

# shapefile

counties = gpd.read_file(
    "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
)

counties = counties.rename(columns={"GEOID": "county_fips"})


gdf = counties.merge(
    df,
    on="county_fips",
    how="left"
)

# if county has an estimate

gdf["has_gap"] = gdf[GAP_COL].notna()

vmin = gdf.loc[gdf["has_gap"], GAP_COL].min()
vmax = gdf.loc[gdf["has_gap"], GAP_COL].max()

print(f"Gap range: {vmin:.4f} to {vmax:.4f}")

# draw map

def plot_state_map(state_abbrev, output_file):

    state = gdf[gdf["STATEFP"] == {
        "CA": "06",
        "TX": "48"
    }[state_abbrev]].copy()

    fig, ax = plt.subplots(figsize=(12, 10))

    # if county does not have enough data

    state[~state["has_gap"]].plot(
        color="#d9d9d9",
        edgecolor="white",
        linewidth=0.3,
        ax=ax
    )

    # counties that do have valid estimates

    state[state["has_gap"]].plot(
        column=GAP_COL,
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        linewidth=0.3,
        edgecolor="white",
        legend=True,
        legend_kwds={
            "label": "Approval Gap (percentage points)",
            "shrink": 0.7
        },
        ax=ax
    )

    # State outline
    state.boundary.plot(
        ax=ax,
        color="black",
        linewidth=0.8
    )

    # legend

    missing_patch = Patch(
        facecolor="#d9d9d9",
        edgecolor="black",
        label="Gap Not Estimated\n(Sample Threshold Not Met)"
    )

    ax.legend(
        handles=[missing_patch],
        loc="lower left",
        frameon=True
    )

    ax.set_title(
        f"{state_abbrev} County Approval Disparity Gap",
        fontsize=18,
        pad=15
    )

    ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved {output_file}")


plot_state_map(
    state_abbrev="TX",
    output_file="tx_approval_gap_map.png"
)

plot_state_map(
    state_abbrev="CA",
    output_file="ca_approval_gap_map.png"
)

# top 25 table

top25 = (
    gdf[gdf["has_gap"]]
    .sort_values(GAP_COL, ascending=False)
    .head(25)
    .copy()
)

top25["County"] = top25["NAME"]

# Rank
top25["Rank"] = range(1, len(top25) + 1)

table_df = top25[
    [
        "Rank",
        "County",
        STATE_COL,
        "county_fips",
        GAP_COL
    ]
].copy()

table_df.columns = [
    "Rank",
    "County",
    "State",
    "FIPS",
    "Approval Gap"
]

table_df["Approval Gap"] = (
    table_df["Approval Gap"]
    .round(4)
)


fig_height = max(8, len(table_df) * 0.45)

fig, ax = plt.subplots(
    figsize=(11, fig_height)
)

ax.axis("off")

tbl = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc="center",
    cellLoc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.4)

# Header formatting
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_height(0.06)

plt.title(
    "Top 25 Counties by Approval Disparity Gap",
    fontsize=18,
    pad=20
)

plt.savefig(
    "top25_approval_gap_table.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved top25_approval_gap_table.png")

