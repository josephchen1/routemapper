# ✈️ Flight Map Generator

Interactive flight route map generator built with Streamlit + Cartopy.

## Run locally

```bash
pip install -r requirements.txt
streamlit run flight_map_app.py
```

## Files

| File | Required | Notes |
|------|----------|-------|
| `routes2.csv` | Yes | Upload in app |
| `airport_coords.csv` | Yes | Upload in app |

## CSV format

**routes2.csv**: `origin, dest, color, style, linewidth, dotsize, dotcolor, connector, labelsize, zorder`

**airport_coords.csv**: `iata, longitude, latitude, region`

Valid regions: `Domestic`, `Pacific`, `Atlantic`, `South`, `Mexico`, `Central`, `Caribbean`, `Other`
