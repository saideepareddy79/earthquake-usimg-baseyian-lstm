from flask import Flask, render_template, request
from models import run_all_models
import pandas as pd

app = Flask(__name__)
DATA_PATH = "data/EarthquakeData.csv"

@app.route("/", methods=["GET", "POST"])
def index():
    forecast_results = None
    forecast_ranges = None
    error = None

    if request.method == "POST":
        region = request.form.get("region")
        try:
            # Load dataset
            df = pd.read_csv(DATA_PATH)

            # Ensure 'place' exists
            if 'place' not in df.columns:
                raise ValueError("Dataset does not contain 'place' column")

            # Filter by region
            if region:
                df_region = df[df['place'].astype(str).str.contains(region, case=False, na=False)]
                if df_region.empty:
                    raise ValueError(f"No earthquake data found for region '{region}'")
            else:
                df_region = df

            # Run models on filtered dataframe
            results, forecast_ranges, _ = run_all_models(df_region)
            forecast_results = results

        except ValueError as ve:
            error = str(ve)
        except Exception as e:
            error = "An unexpected error occurred: " + str(e)

    return render_template(
        "index.html",
        results=forecast_results,
        forecast_ranges=forecast_ranges,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
