# Dubai Real Estate Price Prediction

An artificial neural network (TensorFlow 2 / Keras) that estimates the sale price of Dubai resale properties, plus **DXB Price**, a website that puts the trained model to use.

Course project for **Programming for Data Analytics II**, Ajman University, Fall 2026.

**Live site:** [dxbprice.me](https://dxbprice.me)

## Results

Results are for the final model (*Final A*) on the **test set**: 7,500 properties that were never used for training, tuning or early stopping.

| Metric | Final A (test) |
|---|---|
| MAPE | **8.8%** (about 91.2% average accuracy) |
| R² | **0.980** |
| MAE | **$85,054** |
| RMSE | **$162,490** |
| Within ±10% / ±20% | 64.3% / 93.2% of properties |

The baseline network had a MAPE of 11.6% and R² 0.967 on validation. Against the baseline, the final model cut MAE by 27% and RMSE by 30%.

Accuracy climbed in stages:

- 20.6% with linear regression
- 46.5% with the first network on raw features
- 88.4% after scaling, the log target and one-hot encoding
- 91.2% after tuning

## Dataset

The dataset is `secondary_sales.csv`: 50,000 Dubai resale listings from 2020 to 2026, with 29 columns covering:

- property type, bedrooms, size, floor and building age
- community, zone and coordinates
- view, furnishing, condition and parking
- metro access and distance to Burj Khalifa
- the mortgage rate at the time of listing

**Target:** `price_usd`, modelled as `log(price)`.

**Removed to prevent target leakage:**

- `price_per_sqft_usd` and `price_per_m2_usd`, which are computed from the price
- the AED price
- `id`
- `area_m2`, which duplicates `area_sqft`

**Cleaning:**

- villas get `floor = 0`
- the listing date is split into year and month

## Method

1. **Split:** 70 / 15 / 15 train / validation / test, with seed 42.
2. **Preprocessing:** a scikit-learn `ColumnTransformer` that:
   - one-hot encodes the 10 categorical columns
   - standardises the 13 numeric columns
   - passes the 2 boolean columns through unchanged

   This gives 235 inputs. The transformer is fitted on the training set only.
3. **Model:** `235 → Dense(64, sigmoid) → Dense(32, sigmoid) → Dense(1, linear)`, 17,217 parameters.
   - Adam optimiser with learning rate 0.001
   - MSE loss on log(price)
   - early stopping with patience 10
4. **Experiments:** 34 controlled training runs, each changing one setting from the baseline. The settings tested:
   - architecture, activation and learning rate
   - optimiser, batch size and epochs
   - L2 and Dropout

   Run-to-run noise was measured first with 4 seeds, so gains smaller than that noise were not counted as wins. Every run is logged in `experiment_log.json`.
5. **Evaluation:** MAE, RMSE, R² and MAPE, computed in US dollars after converting predictions back from log scale.

## Repository layout

```
pda2_project_final.ipynb   full notebook (EDA, preprocessing, experiments, final model), with outputs
experiment_log.json        training curves and metrics for every experiment
secondary_sales.csv        dataset

app.py                     Flask web app: loads the model + preprocessor and serves the API
templates/index.html       the web page (Estimate, Find homes, Sell, maps, 3D city)
metro.py                   downloads Dubai Metro lines once from OpenStreetMap  -> metro_cache.json
photos.py                  looks up community photos on Wikimedia Commons     -> photos_cache_v2.json
dubai_price_model.keras    trained network (Final A)
preprocessor.joblib        fitted one-hot encoder + scaler
requirements.txt           pinned versions the model was saved with
static/                    fonts, Leaflet, Three.js, jsPDF, html-to-image (with their licences)

site/                      static build deployed to dxbprice.me (no server needed)
  index.html               the same page with the dropdown options built in
  api.js                   replaces the Flask API in the browser and runs the network in JavaScript
  api/data.json            network weights, encoder categories, scaler mean/scale, lookups, dataset rows
  api/photos.json, api/metro.json
```

## Run the notebook

Open `pda2_project_final.ipynb` in Jupyter or Google Colab, with `secondary_sales.csv` in the same folder, and run all cells. Running it saves:

- `dubai_price_model.keras`
- `preprocessor.joblib`
- `experiment_log.json`

## Run the website locally

Use Python 3.12. `scikit-learn` must be **1.6.1**, because the preprocessor was saved with that version.

```bash
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000>.

The map uses Google Maps when a browser API key is placed in a file called `maps_key.txt` next to `app.py`. That file is not in this repository and is ignored by git. Without it, the map falls back to Leaflet with OpenStreetMap.

## How the website uses the model

- **Flask version:** `app.py` builds the 25 input features for a property, filling location details from community lookups. It transforms them with `preprocessor.joblib`, runs the network, then applies `exp()` to the output to get the price back in dollars.
- **Static version:** the live site (`site/`) has no server. `api.js` intercepts the page's API calls and performs the same forward pass in JavaScript, using the exported weights and scaler values in `api/data.json`.

The `site/` folder is hosted as a static site on Cloudflare.
