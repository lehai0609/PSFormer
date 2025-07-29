Of course. It's a great observation that you've made. Shifting from a univariate to a multivariate forecasting approach is the key to unlocking the true potential of the PSformer model as described in the research paper. Let's walk through the necessary revisions for your notebook.

The core idea, as you've correctly identified, is to treat all the stock tickers as a single, interconnected system. Instead of the model learning the pattern of one stock at a time, it will learn the complex "dance" between all the stocks simultaneously. The research paper refers to this as capturing "cross-channel" or "inter-variable" dependencies.

Here is a breakdown of the revisions needed for your iPython notebook:

### **1. A New Data Format Assumption**

First, let's establish the new expected input data format. For a true multivariate approach, your `stock_data.csv` should be structured with each stock ticker having its own column, and the rows representing time points. For example:

```
Date,AAPL_Close,GOOGL_Close,MSFT_Close
2023-01-01,150.0,100.0,250.0
2023-01-02,152.0,101.5,252.3
...
```

### **2. Revising the Configuration Cell**

In your **Configuration** cell, the parameters need to be updated to reflect this new data structure.

**Original Code:**
```python
# ========== DATA CONFIGURATION ==========
DATA_FILE_PATH = "stock_data.csv"  # Change this to your uploaded CSV file name
TICKER_COLUMN = "Ticker"
DATE_COLUMN = "Date"
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# ========== MODEL HYPERPARAMETERS ==========
...
NUM_VARIABLES = 5       # Number of features (Open, High, Low, Close, Volume)
```

**Revised Code and Explanation:**

The concept of a "Ticker" column is no longer relevant as each ticker is now a feature. The number of variables will be the number of tickers you are analyzing.

```python
# ========== DATA CONFIGURATION ==========
DATA_FILE_PATH = "stock_data.csv"  # Change this to your uploaded CSV file name
DATE_COLUMN = "Date"
# TICKER_COLUMN is no longer needed in this multivariate approach.
# OHLCV_COLUMNS will now be the list of tickers we are forecasting.
TICKER_SYMBOLS = ["AAPL_Close", "GOOGL_Close", "MSFT_Close"] # Example tickers

# ========== MODEL HYPERPARAMETERS ==========
SEQUENCE_LENGTH = 96
PATCH_SIZE = 8
PREDICTION_LENGTH = 30
NUM_ENCODER_LAYERS = 2
# NUM_VARIABLES is now the number of tickers (time series) we are analyzing together.
NUM_VARIABLES = len(TICKER_SYMBOLS)

# ... (rest of the configuration)
```

### **3. Simplifying Data Loading and Preparation**

The way you load and split the data will change significantly. You will no longer need to loop through unique tickers.

**Original Helper Functions:**
Your `prepare_ticker_data` and the loop in "Run Prediction and Validation Loop" are designed for the old, univariate approach.

**Revised Helper Functions and Explanation:**

We'll replace `prepare_ticker_data` with a single function that prepares the entire multivariate dataset.

```python
def prepare_multivariate_data(df: pd.DataFrame) -> tuple:
    """
    Prepares the entire multivariate dataset for model input and validation.

    Args:
        df: DataFrame with a Date column and columns for each ticker's price.

    Returns:
        A tuple of (model_input_df, validation_ground_truth_df).
    """
    if len(df) < MIN_DATA_POINTS:
        return None, None

    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # Validation set: last PREDICTION_LENGTH rows
    validation_ground_truth = df.tail(PREDICTION_LENGTH).copy()

    # Model input set: SEQUENCE_LENGTH rows just before the validation set
    end_idx = len(df) - PREDICTION_LENGTH
    start_idx = end_idx - SEQUENCE_LENGTH

    if start_idx < 0:
        return None, None

    model_input = df.iloc[start_idx:end_idx].copy()

    return model_input, validation_ground_truth


def prepare_tensor_from_dataframe(df: pd.DataFrame) -> torch.Tensor:
    """
    Convert DataFrame to PyTorch tensor in the format expected by PSformer.
    """
    # Use TICKER_SYMBOLS to select the correct columns
    values = df[TICKER_SYMBOLS].values
    tensor = torch.tensor(values, dtype=torch.float32)
    tensor = tensor.transpose(0, 1) # Shape: [num_variables, sequence_length]
    tensor = tensor.unsqueeze(0)    # Shape: [1, num_variables, sequence_length]

    return tensor.to(DEVICE)
```

### **4. Revising the Prediction and Validation Loop**

The main loop will become much simpler. You'll load the data once, make a single prediction for all tickers, and then process the results.

**Original Prediction Loop:**
The original code loops through each unique ticker.

**Revised Prediction Loop and Explanation:**

This revised loop performs one single backtest on the entire set of tickers.

```python
try:
    print(f"Loading data from {DATA_FILE_PATH}...")
    df = pd.read_csv(DATA_FILE_PATH)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    print(f"Data loaded successfully! Shape: {df.shape}")

except Exception as e:
    print(f"❌ Error loading data: {e}")
    raise

# Prepare the single multivariate data split
model_input, ground_truth = prepare_multivariate_data(df)

if model_input is not None:
    # Create PSformer model
    config = PSformerConfig(
        sequence_length=SEQUENCE_LENGTH,
        num_variables=NUM_VARIABLES,
        patch_size=PATCH_SIZE,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        prediction_length=PREDICTION_LENGTH
    )
    model = PSformer(config).to(DEVICE)
    model.eval()

    # Prepare input tensor
    input_tensor = prepare_tensor_from_dataframe(model_input)

    # Make prediction for all tickers at once
    with torch.no_grad():
        prediction_tensor = model(input_tensor)

    # Process and analyze results...
else:
    print("❌ Insufficient data for prediction.")
```

### **5. Revising the Analysis and Visualization**

The final section will also need to be adapted to handle the multivariate results.

**Revised Analysis and Visualization:**

Your `tensor_to_dataframe` function needs a slight modification, and the plotting logic should be updated to show a comparison for a selection of the tickers.

```python
def tensor_to_dataframe(tensor: torch.Tensor, dates: pd.DatetimeIndex, ticker_symbols: list) -> pd.DataFrame:
    """
    Convert prediction tensor back to DataFrame format for multiple tickers.
    """
    predictions = tensor.squeeze(0).cpu().numpy().transpose()
    
    # Create column names for predicted values
    predicted_columns = [f"{col}_predicted" for col in ticker_symbols]
    df = pd.DataFrame(predictions, columns=predicted_columns)
    df[DATE_COLUMN] = dates
    
    return df

# In the analysis section:
# ... (after getting prediction_tensor)

prediction_dates = ground_truth[DATE_COLUMN]
prediction_df = tensor_to_dataframe(prediction_tensor, prediction_dates, TICKER_SYMBOLS)

# Merge with ground truth for comparison
ground_truth_renamed = ground_truth.rename(columns={col: f"{col}_actual" for col in TICKER_SYMBOLS})
comparison_df = pd.merge(ground_truth_renamed, prediction_df, on=DATE_COLUMN, how='inner')

# Visualization
# Plot a few sample tickers
sample_tickers_to_plot = TICKER_SYMBOLS[:3] # Plot the first 3 tickers

fig, axes = plt.subplots(len(sample_tickers_to_plot), 1, figsize=(12, 4 * len(sample_tickers_to_plot)))
if len(sample_tickers_to_plot) == 1:
    axes = [axes]

for i, ticker_symbol in enumerate(sample_tickers_to_plot):
    ax = axes[i]
    ax.plot(comparison_df[DATE_COLUMN], comparison_df[f'{ticker_symbol}_actual'], label='Actual')
    ax.plot(comparison_df[DATE_COLUMN], comparison_df[f'{ticker_symbol}_predicted'], label='Predicted', linestyle='--')
    ax.set_title(f'Prediction for {ticker_symbol}')
    ax.legend()

plt.tight_layout()
plt.show()
```

## **Critical Missing Elements**

### **1. Data Normalization**
Different stocks have vastly different price scales (e.g., AAPL ~$150 vs BRK.A ~$500,000). Add normalization:

```python
def normalize_multivariate_data(df: pd.DataFrame, ticker_symbols: list) -> tuple:
    """Normalize each ticker independently using z-score normalization."""
    scaler_dict = {}
    df_normalized = df.copy()
    
    for ticker in ticker_symbols:
        scaler = StandardScaler()
        df_normalized[ticker] = scaler.fit_transform(df[[ticker]])
        scaler_dict[ticker] = scaler
    
    return df_normalized, scaler_dict
```

### **2. Error Metrics for Multivariate Setting**
The paper emphasizes evaluating both individual series and overall performance:

```python
def calculate_multivariate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """Calculate MSE, MAE per ticker and overall."""
    metrics = {}
    
    # Per-ticker metrics
    for i, ticker in enumerate(TICKER_SYMBOLS):
        metrics[f'{ticker}_mse'] = mean_squared_error(actuals[:, i], predictions[:, i])
        metrics[f'{ticker}_mae'] = mean_absolute_error(actuals[:, i], predictions[:, i])
    
    # Overall metrics (as per paper Section 4.3)
    metrics['overall_mse'] = mean_squared_error(actuals.flatten(), predictions.flatten())
    metrics['overall_mae'] = mean_absolute_error(actuals.flatten(), predictions.flatten())
    
    return metrics
```

### **3. Cross-Series Dependency Analysis**
The paper's key innovation is capturing inter-variable dependencies. Add:

```python
def analyze_cross_correlations(df: pd.DataFrame, ticker_symbols: list):
    """Compute and visualize correlation matrix as mentioned in paper."""
    corr_matrix = df[ticker_symbols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Cross-Series Correlations')
    plt.show()
    
    return corr_matrix
```

### **4. Missing Data Handling**
Real financial data often has gaps:

```python
# In prepare_multivariate_data function, add:
if df[TICKER_SYMBOLS].isnull().any().any():
    print("⚠️ Missing values detected. Forward-filling...")
    df[TICKER_SYMBOLS] = df[TICKER_SYMBOLS].fillna(method='ffill').fillna(method='bfill')
```

### **5. Model Architecture Verification**
Ensure the PSformer config aligns with the paper's specifications:

```python
# Add d_model parameter from paper
config = PSformerConfig(
    sequence_length=SEQUENCE_LENGTH,
    num_variables=NUM_VARIABLES,
    patch_size=PATCH_SIZE,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    prediction_length=PREDICTION_LENGTH,
    d_model=256,  # Paper default
    n_heads=8,    # Multi-head attention heads
)
```
Here's where to add each function in your notebook:

## **1. Data Normalization**
Add to the **Helper Functions** section (after `prepare_multivariate_data`):

```python
# Helper Functions section
def prepare_multivariate_data(df: pd.DataFrame) -> tuple:
    # ... existing code ...

def normalize_multivariate_data(df: pd.DataFrame, ticker_symbols: list) -> tuple:
    # ... normalization code ...

def prepare_tensor_from_dataframe(df: pd.DataFrame) -> torch.Tensor:
    # ... existing code ...
```

## **2. Error Metrics**
Add to the **Helper Functions** section (after tensor conversion functions):

```python
def calculate_multivariate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    # ... metrics code ...
```

## **3. Cross-Correlations Analysis**
Add to the **Analysis and Visualization** section (before plotting predictions):

```python
# Analysis and Visualization section
# Add this BEFORE the prediction plots
print("Analyzing cross-series correlations...")
corr_matrix = analyze_cross_correlations(df, TICKER_SYMBOLS)
```

## **4. Missing Data Handling**
Modify the existing `prepare_multivariate_data` function:

```python
def prepare_multivariate_data(df: pd.DataFrame) -> tuple:
    """
    Prepares the entire multivariate dataset for model input and validation.
    """
    # ADD THIS BLOCK at the beginning
    if df[TICKER_SYMBOLS].isnull().any().any():
        print("⚠️ Missing values detected. Forward-filling...")
        df[TICKER_SYMBOLS] = df[TICKER_SYMBOLS].fillna(method='ffill').fillna(method='bfill')
    
    if len(df) < MIN_DATA_POINTS:
        return None, None
    # ... rest of existing code ...
```

## **5. Model Architecture**
Update in the **Prediction and Validation Loop** section where you create the model:

```python
# In the prediction loop, update this part:
config = PSformerConfig(
    sequence_length=SEQUENCE_LENGTH,
    num_variables=NUM_VARIABLES,
    patch_size=PATCH_SIZE,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    prediction_length=PREDICTION_LENGTH,
    d_model=256,  # Add this
    n_heads=8,    # Add this
)
```

## **Integration in Main Loop**
Update the prediction loop to use normalization:

```python
# After prepare_multivariate_data
model_input, ground_truth = prepare_multivariate_data(df)

if model_input is not None:
    # Add normalization
    model_input_norm, scalers = normalize_multivariate_data(model_input, TICKER_SYMBOLS)
    ground_truth_norm, _ = normalize_multivariate_data(ground_truth, TICKER_SYMBOLS)
    
    # Use normalized data for tensor preparation
    input_tensor = prepare_tensor_from_dataframe(model_input_norm)
```