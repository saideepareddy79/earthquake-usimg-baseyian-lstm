import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization

tf.random.set_seed(42)
np.random.seed(42)

# -------------------------
# Helper functions
# -------------------------
def regression_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    data_range = np.max(y_true) - np.min(y_true)
    accuracy = max(0, 100 * (1 - rmse / (data_range + 1e-8)))
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Accuracy (%)": accuracy}

def create_sequences(features_array, target_index=0, seq_len=30):
    X_seq, y_seq = [], []
    for i in range(len(features_array)-seq_len):
        X_seq.append(features_array[i:i+seq_len])
        y_seq.append(features_array[i+seq_len, target_index])
    return np.array(X_seq), np.array(y_seq)

# -------------------------
# Main function
# -------------------------
def run_all_models(df_input):
    results = {}
    forecast_ranges = {}
    trained_models = {}

    df = df_input.copy()

    # Check required columns
    required_cols = ['magnitude', 'time', 'depth_km', 'longitude', 'latitude']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in dataset")

    # Preprocessing
    df = df.dropna(subset=['magnitude']).sort_values('time').reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'])
    df['dayofyear'] = df['time'].dt.dayofyear
    df['hour'] = df['time'].dt.hour

    # Features
    features = ['depth_km','latitude','longitude','dayofyear','hour']
    for f in features:
        if f not in df.columns:
            df[f] = 0.0

    X = df[features].fillna(0)
    y = df['magnitude']

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    # -------------------
    # Random Forest
    # -------------------
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results['Random Forest'] = regression_metrics(y_test, y_pred)
    forecast_ranges['Random Forest'] = (float(y_pred.min()), float(y_pred.max()))
    trained_models['Random Forest'] = rf

    # -------------------
    # Decision Tree
    # -------------------
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    results['Decision Tree'] = regression_metrics(y_test, y_pred)
    forecast_ranges['Decision Tree'] = (float(y_pred.min()), float(y_pred.max()))
    trained_models['Decision Tree'] = dt

    # -------------------
    # Linear Regression
    # -------------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results['Linear Regression'] = regression_metrics(y_test, y_pred)
    forecast_ranges['Linear Regression'] = (float(y_pred.min()), float(y_pred.max()))
    trained_models['Linear Regression'] = lr

    # -------------------
    # SVR
    # -------------------
    svr = SVR()
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    results['SVR'] = regression_metrics(y_test, y_pred)
    forecast_ranges['SVR'] = (float(y_pred.min()), float(y_pred.max()))
    trained_models['SVR'] = svr

    # -------------------
    # Exponential Smoothing
    # -------------------
    y_series = y.values
    split_idx = int(0.8*len(y_series))
    ses_model = ExponentialSmoothing(y_series[:split_idx], seasonal='add', seasonal_periods=7).fit()
    ses_forecast = ses_model.forecast(len(y_series[split_idx:]))
    results['Exponential Smoothing'] = regression_metrics(y_series[split_idx:], ses_forecast)
    forecast_ranges['Exponential Smoothing'] = (float(ses_forecast.min()), float(ses_forecast.max()))
    trained_models['Exponential Smoothing'] = ses_model

    # -------------------
    # Prophet
    # -------------------
    prophet_df = df[['time','magnitude']].rename(columns={'time':'ds','magnitude':'y'})
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(prophet_df.iloc[:split_idx])
    future = prophet_model.make_future_dataframe(periods=len(y_series[split_idx:]), freq='D')
    forecast = prophet_model.predict(future)
    prophet_pred = forecast['yhat'][-len(y_series[split_idx:]):].values
    results['Prophet'] = regression_metrics(y_series[split_idx:], prophet_pred)
    forecast_ranges['Prophet'] = (float(prophet_pred.min()), float(prophet_pred.max()))
    trained_models['Prophet'] = prophet_model

    # -------------------
    # LSTM
    # -------------------
    feature_cols = ['magnitude','depth_km','latitude','longitude','dayofyear','hour']
    feature_data = df[feature_cols].values.astype(float)
    scaler_lstm = MinMaxScaler()
    feature_scaled = scaler_lstm.fit_transform(feature_data)
    X_seq, y_seq = create_sequences(feature_scaled, target_index=0, seq_len=30)
    split = int(0.8*len(X_seq))
    X_train_lstm, X_test_lstm = X_seq[:split], X_seq[split:]
    y_train_lstm, y_test_lstm = y_seq[:split], y_seq[split:]

    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=25, batch_size=32, verbose=0)
    y_pred = lstm_model.predict(X_test_lstm).flatten()
    y_pred_inv = scaler_lstm.inverse_transform(
        np.hstack([y_pred.reshape(-1,1), np.zeros((len(y_pred), len(feature_cols)-1))])
    )[:,0]
    y_test_inv = scaler_lstm.inverse_transform(
        np.hstack([y_test_lstm.reshape(-1,1), np.zeros((len(y_test_lstm), len(feature_cols)-1))])
    )[:,0]
    results['LSTM'] = regression_metrics(y_test_inv, y_pred_inv)
    forecast_ranges['LSTM'] = (float(y_pred_inv.min()), float(y_pred_inv.max()))
    trained_models['LSTM'] = lstm_model

    # -------------------
    # Bayesian LSTM
    # -------------------
    inputs = Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]))
    x = LSTM(128, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x, training=True)
    x = LSTM(64, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x, training=True)
    x = LSTM(32)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x, training=True)
    outputs = Dense(1)(x)
    bayesian_model = Model(inputs, outputs)
    bayesian_model.compile(optimizer='adam', loss='mse')
    bayesian_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)

    MC_SAMPLES = 50
    mc_preds = np.array([bayesian_model(X_test_lstm, training=True).numpy().flatten() for _ in range(MC_SAMPLES)])
    mc_mean = mc_preds.mean(axis=0)
    mc_std = mc_preds.std(axis=0)
    mc_mean_inv = scaler_lstm.inverse_transform(
        np.hstack([mc_mean.reshape(-1,1), np.zeros((len(mc_mean), len(feature_cols)-1))])
    )[:,0]
    mc_std_inv = mc_std * (scaler_lstm.data_max_[0]-scaler_lstm.data_min_[0])
    results['Bayesian LSTM'] = regression_metrics(y_test_inv, mc_mean_inv)
    results['Bayesian LSTM']['Uncertainty (std)'] = float(np.mean(mc_std_inv))
    forecast_ranges['Bayesian LSTM'] = (float(mc_mean_inv.min()), float(mc_mean_inv.max()))
    trained_models['Bayesian LSTM'] = bayesian_model

    return results, forecast_ranges, trained_models
