# Predictions of Groundwater Arsenic Concentrations using Graph Neural Networks

## Project Overview
In this project, we use different machine learning models including random forests, neural networks, gradient boosted trees, and a variety of graph neural networks to predict the amount of arsenic contamination in water across the continental United States. To improve predictive performance, we design our own comprehensive dataset containing 74,706 samples and 115 features derived from multiple public data sources. These features capture critical geological, hydrological, and environmental factors that collectively enhance the accuracy of arsenic contamination prediction.

## Data Sources
To develop a unified, feature-rich, and spatially consistent dataset for arsenic contamination prediction, we integrated information from multiple large-scale public repositories covering water quality, soil chemistry, hydrology, land cover, and geological context across the continental United States. Each dataset contributes complementary environmental signals that correlate with underlying arsenic mobilization processes, including redox chemistry, sediment composition, aquifer depth, land use, and surface hydrology.

1. Water Quality Portal ([WQP – EPA, USGS, and NOAA](https://www.waterqualitydata.us/))

The WQP contains well-level arsenic measurements, metadata about sampling frequency, and accompanying water chemistry parameters. This dataset serves as the primary source of ground truth labels for our target variable: arsenic concentration (µg/L). After filtering and quality-control processing, we retained laboratory-verified arsenic observations along with spatial coordinates and temporal sampling metadata.

**Purpose in the model**: Defines the regression target and provides geolocated observations for supervised learning and geospatial modeling.

**Direct Download Instructions**: View the link in the title. For country, select "United States". **Incomplete**

2. USGS Mineral Resources Data System ([MRDS](https://mrdata.usgs.gov/mrds/))

The MRDS characterizes geological provinces, mineral deposits, ore bodies, mining districts, and regional bedrock composition. We extracted categorical and numeric features related to ore type, deposit geology, land status, production history, and material composition. These features are strong predictors because arsenic is commonly mobilized from sulfide minerals, sedimentary formations, and geothermal zones.

**Purpose in the model**: Adds geological context and provides coarse spatial priors for arsenic mobilization risk.

**Direct Download Instructions**: View the link in the title and download this file: "rdbms-tab-all.zip".

3. gNATSGO Soil Chemistry and Hydrology Dataset (https://nrcs.app.box.com/v/soils/folder/233393842838)

The gNATSGO database provides detailed soil chemical composition, organic matter content, clay fraction, pH, depth-to-restrictive layer, drainage class, and hydrologic properties at high spatial resolution (10–30 m to 250 m depending on layers). We spatially interpolated soil attributes to each WQP sampling coordinate using geodesic nearest-neighbor extraction.

**Purpose in the model**: Constrains subsurface arsenic mobility via sediment composition, grain size, redox buffering capacity, organic carbon content, and aquifer recharge conditions.

**Direct Download Instructions**: View the link in the title and download this file: "gNATSGO_02_03_2025.7z".

## Training Dataset Creation Pipeline

The final training dataset is engineered by independently cleaning three major data sources (WQP, MRDS, and gNATSGO), enriching each sample with geological and soil attributes, and merging all processed layers into a unified prediction-ready table.

---

### **1. WQP Cleaning**

**File:** `WQP Cleaning.py`  
**Purpose:** Cleans raw water quality measurements by:

- Filtering for valid arsenic samples
- Standardizing numeric fields and geographic coordinates
- Removing malformed values and duplicated records

**Output:** A clean WQP sample table containing only high-quality arsenic measurements with accurate lat/lon values.

---

### **2. MRDS Preprocessing and Cleaning**

#### **Step A — Convert TXT to CSV**

**File:** `MRDS_txt_to_csv.py`  
Converts raw MRDS `.txt` attribute tables into consistent `.csv` files while:

- Preserving empty fields
- Normalizing spacing
- Removing quote artifacts

#### **Step B — Merge MRDS Attribute Tables**

**File:** `MRDS Merged Creation.py`  
Aggregates all MRDS tables using `dep_id` and consolidates multi-valued attributes into single unified rows.

#### **Step C — Clean MRDS Attributes**

**File:** `MRDS Cleaning.py`  
Cleans the merged MRDS dataset by:

- Dropping highly sparse fields
- Standardizing coordinates and geologic fields
- Normalizing text attributes

#### **Step D — Compute Geological Distance Features**

**File:** `Haversine Merging.py`  
Computes the distance between each WQP sample and its nearest MRDS deposit using the haversine formula, adding new geological proximity features.

**Output:** A WQP + MRDS enriched dataset containing both sample chemistry and contextual geologic information.

---

### **3. gNATSGO Mapping and Cleaning**

#### **Spatial Mapping (External GIS Step)**  
WQP + MRDS sample points are spatially joined to gNATSGO soil raster layers using GIS software (e.g., QGIS or ArcGIS). Extracted attributes include:

- Soil chemistry
- Hydrologic factors
- Permeability
- Organic matter
- Other environmental soil variables

#### **Cleaning Step**

**File:** `gNATSGO Cleaning.py`  
Standardizes soil attributes, removes invalid extractions, and prepares soil features for merging.

**Output:** A WQP + MRDS + gNATSGO feature-rich dataset with geologic and soil layer context.

---

### **4. Final Dataset Assembly**

**File:** `Final Dataset Creation.py`  
Merges all processed layers into one unified table:

- Harmonizes field types and units
- Removes invalid or mismatched rows
- Ensures all samples have complete location, geologic, and soil attributes

---

### Final Output Summary

- **Rows:** 74,706 samples  
- **Features:** 115 engineered predictors  
- **Sources:** WQP measurements + MRDS geological attributes + gNATSGO soil/hydrology layers  
- **Usage:** Serves as the master dataset for all boosted models and GNN prediction experiments.

## Core Modeling Pipeline

This repository includes four experiment families:

- Gradient Boosted Trees
- Random Forests
- Standard Feedforward Neural Networks
- Multiple Graph Neural Networks (GCN, GCNII, GAT, GraphSAGE, GIN)

All models use the same engineered dataset and identical evaluation metrics, enabling consistent benchmarking across modeling paradigms.

---

### Tree-Based Models

#### `gb_boosted_trees.py`
This file trains several boosted ensemble models, including **XGBoost**, **LightGBM**, and **CatBoost**. Each model is already defined inside the script, but only one runs at a time.

To switch between models, simply **uncomment one block at a time**, for example:

```python
# model = XGBoost(...)
# model = LightGBM(...)
# model = CatBoost(...)
```

Each boosted model uses the same:

- Train/validation/test splitting  
- Standardized evaluation (MAE, MSE, R²)  
- Feature preprocessing  

This allows **direct performance comparison without modifying the pipeline**.

---

#### `random_forest.py`
This script trains a baseline Random Forest model using the same processed dataset. No configuration changes are needed — just run:

```
python random_forest.py
```

The Random Forest serves as a strong baseline for comparison against boosted trees and GNNs.

---

### Fully Connected Neural Network

#### `neural_network.py`
This file trains a standard feedforward neural network (no graph structure). It uses the same dataset filtering, scaling, and evaluation methodology as the tree-based and graph-based models.

To run:

```
python neural_network.py
```

This provides a non-graph neural baseline that models nonlinear relationships in tabular structure only.

---

### Graph Neural Networks

Our spatial models use **k-nearest neighbors over latitude/longitude coordinates** to define edge connectivity. This turns the dataset into a geographical graph where each node represents a sampling site and edges connect geographically nearby points.

---

#### `graph_models.py`
This file contains **all GNN architecture definitions**, including:

- **GCN**
- **GCNII**
- **GAT**
- **GraphSAGE**
- **GIN**

Each model is implemented in a reusable class and is interchangeable with identical preprocessing and evaluation.

You do **not** train models inside this file — you only define them.

---

#### `graph_model_training.py`
This script performs **all GNN experimentation**, including:

- Loading the dataset
- Filtering out non-arsenic parameters
- Standardizing features
- Constructing a spatial neighbor graph using k-nearest neighbors
- Creating the PyTorch Geometric `Data` object
- Training, validating, and testing selected GNN models
- Computing MAE, MSE, and R²
- Plotting predicted vs. actual values and training losses

Inside the file, multiple models are already set up — only one runs at a time.

To switch architectures, simply **uncomment the desired model initialization line**, for example:

```python
# model = GCN(...)
# model = GCNII(...)
# model = GAT(...)
model = GraphSAGE(...)   # currently active
# model = GIN(...)
```

This requires **no changes** to the dataset, masks, or training loop.

---

### Target Scaling Strategies

We evaluated models under two different target-space configurations.

---

#### **1. Logarithmic Scale**
To stabilize skewed arsenic concentration distributions, we transform:

```
y = log(arsenic + 1)
```

Benefits:

- Reduces extreme variance
- Improves neural optimization
- Improves accuracy across all model classes

This was our primary evaluation space.

---

#### **2. Real Concentration Scale**
For interpretability on physical units (µg/L), we also evaluated:

- Using raw arsenic values
- Removing extreme values above **100 µg/L**
- Training and evaluating directly on real concentration scale

This allows comparison between:

- Statistical accuracy in log-scale
- Real-world interpretability of predictions on physical arsenic levels

---

### Running Experiments

Each experiment is standalone and can be executed directly from the root of the project:

```
python gb_boosted_trees.py
python random_forest.py
python neural_network.py
python graph_model_training.py
```

To switch between variants:

- **Boosted trees:** uncomment the desired model block inside `gb_boosted_trees.py`
- **Graph neural networks:** uncomment the desired architecture inside `graph_model_training.py`

No additional configuration is required.

### Our Results
Below shows a table containing our results after running all experiments
![Prediction Results]("Error and Analysis Images/error_results.png")

## Further Result Analysis

After training our core models, all post-training visualization and diagnostic evaluation is performed using the following scripts:

- **`gb_boosted_graph.py`**
- **`graph_model_training.py`**

### What These Scripts Produce

Running these files generates a variety of plots that allow deeper insight into model performance, including:

- **Predicted vs. Actual scatter plots**
- **Train vs. Validation loss curves**
- **Spatial error heatmaps for graph-based models**
- **Residual distributions**
- **Comparison of log-scaled predictions vs. real-scale predictions**

### How To Run

To generate the analysis plots, run:

    python gb_boosted_graph.py
    python graph_model_training.py

### Pipeline Behavior

- No manual data formatting is required  
- Each script loads saved model predictions  
- Evaluation data is automatically formatted  
- All plots are generated and displayed automatically

## Future Work

There are several clear extensions that would further improve model accuracy, scalability, and scientific value:

### **1. Better Spatial Graph Construction**
Our current graph uses K-Nearest Neighbors on latitude/longitude. A stronger method would:
- Use hydrological or watershed connectivity instead of pure distance  
- Add variable-radius spatial neighborhoods rather than fixed K  
- Learn edge weights based on soil chemistry or groundwater flow direction  

This would create a graph structure more aligned with **real water transport behavior**, not just physical proximity.

### **2. Expanded Environmental Feature Layers**
Model performance would likely improve with additional geoscience layers, including:
- Local groundwater pumping density and historical well usage
- High-resolution soil permeability or porosity maps
- Land-use and industrial activity layers

These features would increase the predictive power of both tree models and GNNs by capturing **drivers of arsenic mobility**, not just static chemistry data.

### **3. Unified Multi-Dataset Graph**
Instead of merging WQP, MRDS, and gNATSGO into a flat table, a future model could:
- Represent **each dataset as a different node type**
- Connect geologic nodes, soil nodes, and water sample nodes in a single heterogeneous graph  
- Train a relational GNN to learn contamination from **interacting environmental processes**, rather than only point-level features

This would allow the model to learn **how soil chemistry, mineral resources, and spatial relationships jointly influence contamination**, offering a more scientifically grounded prediction framework.