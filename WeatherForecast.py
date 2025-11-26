from dotenv import load_dotenv
import os

env_path = os.path.join(os.getcwd(), ".env")
load_dotenv(env_path)
import streamlit as st
import os
import pandas as pd
import numpy as np
import requests
import joblib
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from functools import wraps
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# CONFIGURATION

@dataclass
class Config:
    API_KEY: str = os.getenv("API_KEY", "")
    CSV_PATH: Path = Path("weather_data.csv")
    MODEL_PATH: Path = Path("models/weather_model.pkl")
    METRICS_PATH: Path = Path("models/metrics.json")
    LOG_PATH: Path = Path("logs/weather_forecast.log")
    
    N_ESTIMATORS: int = 300
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    
    API_TIMEOUT: int = 10
    MAX_RETRIES: int = 3
    
    def __post_init__(self):
        """Create necessary directories for models and logs."""
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# LOGGING SETUP

def setup_logging(config: Config) -> logging.Logger:
    """Configure comprehensive file and console logging."""
    logger = logging.getLogger("WeatherForecast")

    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(config.LOG_PATH)
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for retrying a function call on failure (e.g., API calls).
    Uses exponential backoff for delay.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator


def log_execution_time(logger: logging.Logger):
    """Decorator to log function execution time for performance monitoring."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} executed in {elapsed:.2f}s")
            return result
        return wrapper
    return decorator

# DATA VALIDATION
class DataValidator:
    """Methods for validating input data and API responses."""
    
    @staticmethod
    def validate_city_name(city: str) -> bool:
        """Validate city name format (simple check)."""
        if not city or not isinstance(city, str):
            return False
        if len(city) < 2 or len(city) > 100:
            return False
        return True
    
    @staticmethod
    def validate_weather_data(data: Dict) -> bool:
        """Validate required keys in the weather API response."""
        required_keys = ["main", "wind"]
        if not all(key in data for key in required_keys):
            return False
        
        required_main = ["humidity", "pressure", "temp"]
        if not all(key in data["main"] for key in required_main):
            return False
        
        if "speed" not in data["wind"]:
            return False
        
        return True
    
    @staticmethod
    def validate_features(features: Dict) -> bool:
        """Validate feature dictionary before prediction."""
        required = ["humidity", "pressure", "wind_speed", "year", "month", "day"]
        return all(key in features for key in required)

# WEATHER API CLIENT

class WeatherAPIClient:
    """Robust weather API client with error handling and retry logic."""
    
    def __init__(self, api_key: str, timeout: int = 10, logger: Optional[logging.Logger] = None):
        self.api_key = api_key
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    @retry_on_failure(max_retries=3)
    def fetch_current_weather(self, city: str) -> Tuple[Dict, float]:
        """
        Fetch current weather data from OpenWeatherMap API.
        
        Args:
            city: City name
            
        Returns:
            Tuple of (features dict, actual temperature)
            
        Raises:
            ValueError: If city is invalid or API returns bad data.
            requests.RequestException: If API call fails after all retries.
        """
        if not DataValidator.validate_city_name(city):
            raise ValueError(f"Invalid city name: {city}")
        
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric"
        }
        
        self.logger.info(f"Fetching weather data for: {city}")
        
        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if not DataValidator.validate_weather_data(data):
                raise ValueError("Invalid API response format")
            
            features = self._extract_features(data, city)
            actual_temp = data["main"]["temp"]
            
            self.logger.info(f"Successfully fetched data for {city}: {actual_temp}°C")
            return features, actual_temp
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise
        except (KeyError, ValueError) as e:
            self.logger.error(f"Data extraction failed: {str(e)}")
            raise ValueError(f"Failed to extract weather data: {str(e)}")
    
    def _extract_features(self, data: Dict, city: str) -> Dict:
        now = datetime.now()

        features = {
            "city": city,
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],

            # same as dataset
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "day_of_year": now.timetuple().tm_yday,
            "quarter": (now.month - 1) // 3 + 1,
            "month_sin": np.sin(2 * np.pi * now.month / 12),
            "month_cos": np.cos(2 * np.pi * now.month / 12),
            "humidity_pressure": data["main"]["humidity"] * data["main"]["pressure"] / 100000
        }

        return features


# MODEL TRAINING AND EVALUATION

def fix_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix and standardize the dataset so that the model can learn properly.
    Includes:
      - proper date parsing
      - correct feature engineering
      - ensuring chronological order
      - ensuring no missing values
    """

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date"]).reset_index(drop=True)

    df = df.sort_values("date")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["quarter"] = df["date"].dt.quarter

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["humidity_pressure"] = (df["humidity"] * df["pressure"]) / 100000

    return df

class WeatherModelTrainer:
    """Handles model training, evaluation, and persistence (MLOps)."""
    
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.feature_columns = None
        self.metrics = {}
    
    @log_execution_time(logging.getLogger(__name__))
    def train_model(self) -> Dict[str, float]:
        """
        Runs the full training workflow: load, engineer features, train, evaluate, and save.
        
        Returns:
            Dictionary of evaluation metrics.
        """
        self.logger.info("Starting model training...")
        
        df = self._load_and_preprocess_data()
        X, y = self._prepare_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )
        
        self.model = self._build_pipeline(X)
        
        self.logger.info(f"Training on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        self.metrics = self._evaluate_model(X_train, X_test, y_train, y_test)
        
        self._save_model()
        self._save_metrics()
        
        self.logger.info(f"Model training complete. Test MAE: {self.metrics['test_mae']:.2f}")
        return self.metrics
    
    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """Load, clean, and FIX dataset."""
        try:
            df = pd.read_csv(self.config.CSV_PATH)
            self.logger.info(f"Loaded {len(df)} records from {self.config.CSV_PATH}")
        except FileNotFoundError:
            self.logger.error(f"Data file not found: {self.config.CSV_PATH}")
            raise
        
        df = df.dropna(subset=["temperature"])

        df = fix_dataset(df)

        self.logger.info("Dataset fixed successfully")
        self.logger.info(f"Final rows: {len(df)}")

        return df
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features (X) and target (y)."""
        X = df.drop(columns=["temperature", "date"], errors="ignore")
        y = df["temperature"]
        
        self.feature_columns = list(X.columns)
        self.logger.info(f"Features: {self.feature_columns}")
        
        return X, y
    
    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Build the full ML pipeline including preprocessing and the regressor."""
        
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if c not in cat_cols]
        
        self.logger.info(f"Categorical features: {cat_cols}")
        self.logger.info(f"Numerical features: {num_cols}")
        
        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols)
        ], remainder='drop')
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=self.config.N_ESTIMATORS,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2
            ))
        ])
        
        return pipeline
    
    def _evaluate_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Comprehensive model evaluation on train, test, and cross-validation sets."""
        
        train_preds = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_preds)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        train_r2 = r2_score(y_train, train_preds)
    
        test_preds = self.model.predict(X_test)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_r2 = r2_score(y_test, test_preds)
        
        cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=self.config.CV_FOLDS,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        metrics = {
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "cv_mae": cv_mae,
            "cv_std": cv_std,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Model Evaluation:\n"
                         f"  Test MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, R²: {test_r2:.3f}\n"
                         f"  CV MAE: {cv_mae:.2f} ± {cv_std:.2f}")
        
        return metrics
    
    def _save_model(self):
        """Save trained model pipeline and feature columns to disk."""
        model_data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "config": asdict(self.config)
        }
        joblib.dump(model_data, self.config.MODEL_PATH)
        self.logger.info(f"Model saved to {self.config.MODEL_PATH}")
    
    def _save_metrics(self):
        """Save evaluation metrics to JSON for monitoring."""
        with open(self.config.METRICS_PATH, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {self.config.METRICS_PATH}")

# MODEL INFERENCE

class WeatherPredictor:
    """Handles model loading and making predictions."""
    
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.feature_columns = None
        self._load_model()
    
    def _load_model(self):
        """Load trained model and feature columns from disk."""
        if not self.config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {self.config.MODEL_PATH}")
        
        try:
            model_data = joblib.load(self.config.MODEL_PATH)
            self.model = model_data["model"]
            self.feature_columns = model_data["feature_columns"]
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict(self, features: Dict) -> float:
        """
        Predict temperature from features.
        
        Args:
            features: Dictionary of feature values.
            
        Returns:
            Predicted temperature in Celsius.
        """
        if not DataValidator.validate_features(features):
            raise ValueError("Invalid features provided")
        
        df = pd.DataFrame([features])
        
        df = df.reindex(columns=self.feature_columns, fill_value=0)
        
        prediction = self.model.predict(df)[0]
        
        self.logger.info(f"Prediction: {prediction:.2f}°C")
        return prediction

# STREAMLIT

def display_metrics(metrics: Dict):
    """Display model metrics in Streamlit."""
    st.subheader("📊 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test MAE", f"{metrics['test_mae']:.2f}°C")
        st.metric("Train MAE", f"{metrics['train_mae']:.2f}°C")
    
    with col2:
        st.metric("Test RMSE", f"{metrics['test_rmse']:.2f}°C")
        st.metric("Test R²", f"{metrics['test_r2']:.3f}")
    
    with col3:
        st.metric("CV MAE", f"{metrics['cv_mae']:.2f}°C")
        st.caption(f"±{metrics['cv_std']:.2f}°C")
    
    with st.expander("📈 Detailed Metrics"):
        st.json(metrics)


def display_prediction_results(
    city: str,
    predicted_temp: float,
    actual_temp: float,
    features: Dict,
    error: float
):
    """Display prediction results in a user-friendly format."""
    st.subheader(f"🌡️ Weather Prediction for {city}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Predicted Temperature",
            f"{predicted_temp:.2f}°C",
            delta=f"{error:.2f}°C" if error else None
        )
    
    with col2:
        st.metric("Actual Temperature", f"{actual_temp:.2f}°C")
    
    with col3:
        accuracy = max(0, 100 - abs(error) * 2)
        st.metric("Prediction Accuracy", f"{accuracy:.1f}%")
    
    with st.expander("🔍 Input Features"):
        feature_df = pd.DataFrame([features]).T
        feature_df.columns = ["Value"]
        feature_df["Value"] = feature_df["Value"].astype(str)
        st.dataframe(feature_df, use_container_width=True)


def main():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Weather Forecasting System",
        page_icon="🌤️",
        layout="wide"
    )
    
    config = Config()
    logger = setup_logging(config)
    
    st.title("🌤️ Weather Forecasting System")

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if st.button("🔄 Retrain Model"):
            with st.spinner("Training model..."):
                try:
                    trainer = WeatherModelTrainer(config, logger)
                    metrics = trainer.train_model()
                    st.success("✅ Model trained successfully!")
                    display_metrics(metrics)
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    logger.error(f"Training error: {str(e)}", exc_info=True)
        
        if config.METRICS_PATH.exists():
            with open(config.METRICS_PATH) as f:
                metrics = json.load(f)
            
            st.divider()
            st.subheader("Current Model Stats")
            st.metric("Test MAE", f"{metrics['test_mae']:.2f}°C")
            st.metric("Test R²", f"{metrics['test_r2']:.3f}")
            st.caption(f"Last trained: {metrics['timestamp'][:16]}")
    
    if not config.MODEL_PATH.exists():
        st.warning("⚠️ No trained model found. Please train a model first.")
        if st.button("Train Initial Model"):
            with st.spinner("Training model for the first time..."):
                try:
                    trainer = WeatherModelTrainer(config, logger)
                    trainer.train_model()
                    st.success("✅ Model trained successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            city = st.text_input(
                "🌍 Enter City Name",
                value="Delhi",
                help="Enter any city name worldwide"
            )
        
        with col2:
            st.write("")
            st.write("")
            predict_button = st.button("🔮 Predict Temperature", type="primary")
        
        if predict_button:
            if not city:
                st.error("Please enter a city name")
            else:
                try:
                    api_client = WeatherAPIClient(config.API_KEY, logger=logger)
                    predictor = WeatherPredictor(config, logger=logger)
                    
                    with st.spinner(f"Fetching weather data for {city}..."):
                        features, actual_temp = api_client.fetch_current_weather(city)
                    
                    with st.spinner("Making prediction..."):
                        predicted_temp = predictor.predict(features)
                    
                    error = predicted_temp - actual_temp
                    
                    display_prediction_results(
                        city, predicted_temp, actual_temp, features, error
                    )
                    
                    logger.info(
                        f"Prediction for {city}: "
                        f"Predicted={predicted_temp:.2f}°C, "
                        f"Actual={actual_temp:.2f}°C, "
                        f"Error={error:.2f}°C"
                    )
                    
                except ValueError as e:
                    st.error(f"❌ Validation Error: {str(e)}")
                    logger.error(f"Validation error: {str(e)}")
                except requests.RequestException as e:
                    st.error(f"❌ API Request Failed: {str(e)}")
                    logger.error(f"API error: {str(e)}", exc_info=True)
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred: {str(e)}")
                    logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        
        st.divider()
        with st.expander("📚 Research & Analysis Tools"):
            st.markdown("""
            ### Available Features
            - **Model Retraining**: Trigger retraining with updated data.
            - **Performance Metrics**: View comprehensive evaluation stats (MAE, RMSE, R²).
            - **Logging**: Full audit trail of actions and errors in `logs/weather_forecast.log`.
            - **Model Versioning**: Metrics include a timestamp for performance comparison over time.
            - **Advanced Features**: Utilizes cyclical encoding for time and interaction features.
            - **Robust Evaluation**: Includes 5-fold cross-validation (CV) results.
            """)


if __name__ == "__main__":
    main()
