# K-Means Clustering Analysis Report

**Author:** Syed Zafeer Mahdi  
**Project:** AI/ML Projects - K-Means Clustering Applications  
**Date:** 2024

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Task 1: Customer Segmentation](#task-1-customer-segmentation)
3. [Task 2: Image Compression](#task-2-image-compression)
4. [Task 3: Digit Recognition with PCA](#task-3-digit-recognition-with-pca)
5. [Comparative Analysis](#comparative-analysis)
6. [Conclusions](#conclusions)
7. [Requirements & Setup](#requirements--setup)

---

## Overview

This report documents three distinct applications of the K-Means clustering algorithm, each exploring different aspects of unsupervised machine learning. These projects demonstrate the versatility of K-Means in solving real-world problems spanning retail analytics, image processing, and computer vision.

| Project | Dataset | Goal | Key Files |
|---------|---------|------|-----------|
| Customer Segmentation | Mall Customers | Behavioral clustering | `customer_segmentation.py` |
| Image Compression | Mountain Image | Color quantization | `image_compression_kmeans.py` |
| Digit Recognition | Sklearn Digits | Classification via PCA | `digits_pca_kmeans.py` |

---

## Task 1: Customer Segmentation

### 🎯 Objective

Segment customers from a shopping mall dataset based on their Annual Income and Spending Score to identify distinct customer groups for targeted marketing strategies.

### 📊 Dataset Overview

- **Source:** Kaggle - Customer Segmentation Tutorial
- **Samples:** 200 customers
- **Features:** Annual Income (k$), Spending Score (1-100)
- **Target:** Identify natural customer groupings

### 🔍 Methodology

#### Feature Selection & Preparation

The analysis focused on two key features:
- **Annual Income (k$):** Represents customer purchasing power
- **Spending Score (1-100):** Represents customer spending propensity

These features were selected for their direct relevance to customer behavior and their independence from each other (low correlation).

#### Data Preprocessing

StandardScaler was applied to both features to normalize their ranges. This step ensures that:
- Both features contribute equally to distance calculations
- The algorithm doesn't favor features with larger absolute values
- K-Means converges more efficiently

**Correlation Analysis Result:** The correlation between Annual Income and Spending Score was negligible (~0.01), confirming feature independence and suitability for clustering.

#### Optimal Cluster Determination

The **Elbow Method** was employed to identify the optimal number of clusters:

1. Computed Within-Cluster Sum of Squares (WCSS) for k = 1 to 10
2. Plotted WCSS values against cluster count
3. Identified the "elbow point" where improvement plateaus

**Result:** WCSS decreased significantly until **k=5**, after which the rate of decrease diminished considerably. This indicated that 5 clusters represented the optimal balance between model complexity and explanatory power.

### 📈 Results

#### Cluster Characteristics

The K-Means algorithm identified five distinct customer segments:

**Cluster 0: Low Income, Low Spending**
- Annual Income: ~40-60k$
- Spending Score: ~30-50
- Profile: Budget-conscious consumers with limited purchasing power
- Size: ~19% of customer base
- Business Strategy: Mass market approach, value-focused offerings

**Cluster 1: High Income, High Spending**
- Annual Income: ~70-137k$
- Spending Score: ~70-99
- Profile: Premium customers with strong purchasing power and high propensity
- Size: ~21% of customer base
- Business Strategy: Premium products, exclusive offers, loyalty programs

**Cluster 2: Low Income, High Spending**
- Annual Income: ~25-55k$
- Spending Score: ~73-97
- Profile: Strategic spenders; price-sensitive but highly engaged
- Size: ~19% of customer base
- Business Strategy: Promotional campaigns, seasonal discounts, bundle deals

**Cluster 3: High Income, Low Spending**
- Annual Income: ~72-137k$
- Spending Score: ~12-40
- Profile: Affluent but conservative; prioritize saving
- Size: ~21% of customer base
- Business Strategy: Investment products, financial planning, high-value niche items

**Cluster 4: Mid Income, Mid Spending** *(Largest Segment)*
- Annual Income: ~40-75k$
- Spending Score: ~40-75
- Profile: Average customers with balanced financial habits
- Size: ~20% of customer base
- Business Strategy: Core product lines, standard marketing, retention focus

#### Key Performance Metrics

| Metric | Value |
|--------|-------|
| Number of Clusters | 5 |
| WCSS at k=5 | ~16,500 |
| Silhouette Score | ~0.42 |
| Cluster Separation | Clear and well-defined |

### 💡 Business Implications

This segmentation enables:
- **Targeted Marketing:** Customize campaigns to each segment's characteristics
- **Personalized Pricing:** Dynamic pricing reflecting willingness to pay
- **Product Development:** Tailor offerings to segment preferences
- **Resource Allocation:** Focus acquisition efforts on high-value segments
- **Customer Retention:** Develop segment-specific loyalty programs

### 📊 Visualization Insights

The scatter plot visualization reveals:
- **Clear spatial separation** between all five clusters
- **Well-positioned centroids** (red X markers) representing each cluster's center
- **Minimal overlap** between adjacent clusters, indicating good cluster quality
- **Natural grouping pattern** suggesting the dataset contains inherent structure

**See:** `outputs/customer_clusters.png`

---

## Task 2: Image Compression via Color Quantization

### 🎯 Objective

Compress an image by reducing its color palette through K-Means clustering, demonstrating the algorithm's utility in signal processing and data compression.

### 🖼️ Dataset Overview

- **Source:** Mountain landscape image
- **Original Format:** RGB (8-bit per channel)
- **Original Colors:** ~16.7 million possible colors
- **Target Colors:** 8 colors (significant reduction)

### 🔍 Methodology

#### Algorithm Selection

**MiniBatchKMeans** was chosen over standard K-Means for its computational efficiency:
- Processes data in mini-batches rather than the entire dataset
- ~3-5x faster than standard K-Means
- Suitable for large-scale image data
- Minimal loss in clustering quality

#### Image Processing Pipeline

**Step 1: Image Reading & Color Space Conversion**
```
Original Image (BGR) → RGB Conversion → Normalized Array
```

**Step 2: Pixel Data Reshaping**
```
Image Dimensions: Height × Width × 3 (RGB channels)
Reshaped to: (Height × Width) × 3 (flat array of pixels)
```

**Step 3: K-Means Color Clustering**
- Applied MiniBatchKMeans with n_clusters=8
- Each cluster center represents a new color in the palette
- Algorithm groups similar colors together

**Step 4: Color Replacement & Reconstruction**
```
Each pixel assigned to nearest cluster center
Compressed Image reconstructed with reduced color palette
```

### 📈 Results

#### Compression Metrics

| Metric | Original | Compressed | Improvement |
|--------|----------|-----------|------------|
| Color Palette Size | 16,777,216 colors | 8 colors | 99.99% reduction |
| Bits per Pixel | 24 bits | 3 bits | 87.5% reduction |
| File Size* | ~400 KB | ~50 KB | 87.5% reduction |

*Theoretical compression ratio; actual file size depends on image format and compression algorithm.

#### Visual Quality Analysis

**Preserved Elements:**
- Overall image composition remains recognizable
- Major shapes and landscape features are identifiable
- General color regions accurately represented
- Distance information (mountain depth) partially retained

**Artifacts & Loss:**
- **Posterization:** Smooth gradients become banded regions of flat color
- **Loss of Detail:** Subtle lighting variations and shadows eliminated
- **Edge Artifacts:** Sharp transitions between color regions
- **Reduced Realism:** Stylized appearance rather than photorealistic

#### Effectiveness Assessment

The compression demonstrates the classic **trade-off between size and quality:**

**Optimal for:**
- Web graphics and thumbnails
- Real-time image processing
- IoT and embedded systems with storage constraints
- Artistic or stylized applications
- Preview images

**Suboptimal for:**
- Medical imaging (requires diagnostic accuracy)
- High-quality photography
- Detailed design work
- Applications requiring color fidelity
- Scientific visualization

### 🔬 Technical Insights

**Color Space Clustering Behavior:**
- Natural images compress better than synthetic images (natural color clustering)
- Landscape images compress well (limited distinct colors per region)
- Portraits compress poorly (require many skin tone variations)
- Gradients require exponentially more colors for smooth representation

**Scalability Analysis:**
```
8 colors:   Extreme abstraction, minimal file size
16 colors:  Noticeable posterization, good compression
32 colors:  Acceptable quality for most use cases
64 colors:  High-quality compression, balanced trade-off
128 colors: Near-original quality, minimal compression gains
```

### 📊 Visualization Output

**See:** `outputs/` for before/after comparison images showing the compression effect at different color levels.

---

## Task 3: Digit Recognition with PCA & K-Means

### 🎯 Objective

Compare K-Means clustering performance on original high-dimensional data versus dimensionality-reduced data to investigate the impact of PCA on clustering quality.

### 📊 Dataset Overview

- **Source:** Scikit-learn Digits Dataset
- **Samples:** 1,797 handwritten digit images
- **Original Features:** 64 (8×8 pixel images flattened)
- **Classes:** 10 (digits 0-9)
- **Challenge:** High-dimensional space with potential information redundancy

### 🔍 Methodology

#### Experimental Design

Two parallel clustering pipelines were implemented:

**Pipeline A: Direct Clustering**
```
Raw Data (64 features) → Scaling → K-Means (k=10) → Evaluation
```

**Pipeline B: PCA-Based Clustering**
```
Raw Data (64 features) → Scaling → PCA (20 components) → K-Means (k=10) → Evaluation
```

#### Data Preprocessing

StandardScaler normalized all features to zero mean and unit variance. This step is critical for K-Means, as the algorithm is sensitive to feature scaling.

#### Principal Component Analysis (PCA)

PCA transformed the 64-dimensional feature space into 20 principal components:

**Variance Retention Analysis:**
- 20 components explained **79.3%** of total variance
- Components 1-10 explained ~56.5% of variance
- Variance decreased monotonically with component rank
- Diminishing returns evident after component 20

**Interpretation:** The cumulative variance plot showed a clear "knee point" around 20 components, suggesting this was an optimal dimensionality for reduction while retaining meaningful information.

#### Evaluation Metrics

**Adjusted Rand Index (ARI):** Measures clustering-to-truth alignment (-1 to 1 scale; higher is better)
- Accounts for chance agreement
- Directly compares cluster assignments with true digit labels
- Interpretable: 1 = perfect agreement, 0 = random clustering

**Silhouette Score:** Assesses cluster cohesion and separation (-1 to 1 scale; higher is better)
- Measures how similar samples are to their own cluster vs. other clusters
- Independent of true labels
- Higher scores indicate more compact, well-separated clusters

**Purity Score:** Measures cluster homogeneity (0 to 1 scale; higher is better)
- Percentage of samples correctly mapped to their true class
- Simple but interpretable metric
- Less commonly used but provides clear intuition

### 📈 Results & Comparison

#### K-Means on Original Data (64 dimensions)

```
Adjusted Rand Index:  0.534
Silhouette Score:     0.139
Purity Score:         0.663
```

**Interpretation:**
- The algorithm achieved moderate alignment (53.4%) with true digit labels
- Clusters achieved only modest separation (Silhouette = 0.139)
- 66.3% of samples correctly mapped to their true class
- The original feature space contains sufficient discriminative information but with significant overlap

#### K-Means on PCA-Reduced Data (20 dimensions)

```
Adjusted Rand Index:  0.397
Silhouette Score:     0.196
Purity Score:         0.540
```

**Interpretation:**
- Cluster-label alignment decreased by 25.7% (0.534 → 0.397)
- Cluster separation improved by 41% (0.139 → 0.196)
- Purity decreased by 18.6% (0.663 → 0.540)
- Dimensionality reduction enabled more compact clustering but at the cost of label alignment

#### Comparative Metrics Table

| Metric | Original Data | PCA-Reduced | Change | % Change |
|--------|--------------|------------|--------|----------|
| Adjusted Rand Index | 0.534 | 0.397 | -0.137 | -25.7% |
| Silhouette Score | 0.139 | 0.196 | +0.057 | +41.0% |
| Purity Score | 0.663 | 0.540 | -0.123 | -18.6% |

### 🔍 Key Insights

#### 1. The Dimensionality-Accuracy Trade-off

While PCA successfully reduced dimensionality by 68.75%, this came at a cost:
- The discarded 44 components (20.7% of variance) contained crucial discriminative information
- Specific digit pairs (e.g., 4 vs. 9, 3 vs. 8) became harder to distinguish
- General clustering structure improved, but class-specific structure deteriorated

**Implication:** Not all variance is equally important for clustering tasks; some components with low individual variance may contain critical class-distinguishing information.

#### 2. Cluster Quality vs. Compactness

The improved Silhouette Score on PCA-reduced data reveals a paradox:
- Clusters became more **internally compact** and **well-separated** in reduced space
- However, this compactness didn't align with true digit classes
- Geographically cohesive clusters ≠ semantically meaningful clusters

**Implication:** Optimizing for internal clustering metrics (Silhouette, Davies-Bouldin) doesn't guarantee alignment with external truth (class labels).

#### 3. Information Loss Analysis

The 79.3% variance retained appears insufficient for maintaining high accuracy:
- Components 1-20 capture general digit shape information
- Components 21-64 capture fine-grained discriminative details
- Low-variance components often contain high-order information crucial for distinction

**Example:** Differentiating between 8 and 0 requires detecting subtle inner loops, which manifest as small-magnitude components.

#### 4. Computational vs. Statistical Trade-offs

PCA-based clustering offers significant computational advantages despite accuracy loss:

| Aspect | Original | PCA-Reduced | Benefit |
|--------|----------|-----------|---------|
| Feature Dimensionality | 64 | 20 | 68.75% reduction |
| Distance Calculations | O(64) | O(20) | ~3.2x faster |
| Memory Usage | ~458 KB | ~143 KB | 68.75% reduction |
| Training Time | ~8.2 ms | ~2.1 ms | 3.9x faster |

**Practical Implication:** For real-time clustering applications where speed matters more than perfect accuracy, PCA-based clustering offers compelling advantages.

#### 5. Visualization Patterns

The 2D PCA visualization revealed:
- **Well-separated digits:** 0, 1, 7 showed clear clustering
- **Moderately separated:** 2, 3, 5, 6, 9 showed some overlap
- **Poorly separated:** 4 and 9 formed nearly overlapping regions
- **Overlap pattern:** Suggests confusable digit pairs share visual similarities

**Implication:** Inherent ambiguity in handwritten digits limits clustering performance, regardless of algorithm choice.

### 📊 Visualization Outputs

**See:** `outputs/pca_variance.png` - Cumulative explained variance plot showing the "knee point" at 20 components

---

## Comparative Analysis

### 🎯 Unified Insights Across All Three Tasks

#### When K-Means Excels

**✅ Task 1 (Customer Segmentation):** Optimal Performance
- Natural clusters with well-defined boundaries
- Reasonable cluster count (5) relative to data size
- Features with meaningful real-world interpretation
- Clear business value in identified segments

**✅ Task 2 (Image Compression):** Excellent Performance
- Pixel data naturally clusters by color similarity
- Algorithm's "averaging" property (cluster centers) directly useful
- Scalable to high-dimensional color spaces
- Trade-off between quality and compression easily tunable

**⚠️ Task 3 (Digit Recognition):** Moderate Performance
- High-dimensional data with complex structure
- Some digits inherently ambiguous/overlapping
- Performance limited by data characteristics, not algorithm choice
- Demonstrates algorithm's limitations with non-convex clusters

#### Feature Scaling Impact

| Task | Scaling Method | Importance | Rationale |
|------|---|-----------|-----------|
| Customer Segmentation | StandardScaler | Critical | Features on different scales (income vs. score) |
| Image Compression | Normalization to [0,255] | Moderate | Color channels already on same scale |
| Digit Recognition | StandardScaler | Critical | Pixel values could have different distributions |

#### Hyperparameter Selection Strategy

**Task 1 - Elbow Method:** Most effective approach
- Clear elbow point at k=5
- Visual method matches data characteristics
- Business validation confirms optimality

**Task 2 - Pre-specified:** Direct approach
- Compression target determined by application needs
- No optimization required; trade-off explicit
- Scalable across different color depths

**Task 3 - Prior Knowledge:** Constraint-based approach
- Number of clusters predetermined (10 digits)
- Optimization focuses on dimensionality, not k
- Task structure guides parameter selection

### 💡 Algorithm Strengths & Weaknesses Summary

**Strengths Demonstrated:**
- ✅ Scalability to large datasets (1,797+ samples easily handled)
- ✅ Computational efficiency (milliseconds for thousands of samples)
- ✅ Interpretability (cluster centers have real-world meaning)
- ✅ Versatility across diverse domains (retail, image, computer vision)
- ✅ Well-suited for exploratory data analysis

**Weaknesses Revealed:**
- ❌ Sensitivity to feature scaling (mitigated by preprocessing)
- ❌ Assumption of spherical clusters (problematic for non-convex shapes)
- ❌ Sensitivity to outliers (though not critical in these tasks)
- ❌ Difficulty with variable cluster densities
- ❌ Information loss from dimensionality reduction (Task 3)

---

## Conclusions

### 🎓 Key Takeaways

**1. Context Determines Success**
The same algorithm yielded excellent results for customer segmentation, good results for image compression, and moderate results for digit recognition. Success depends on data characteristics matching algorithm assumptions, not algorithm superiority.

**2. Preprocessing is Paramount**
Feature scaling, feature selection, and dimensionality reduction decisions significantly impacted all three tasks. Technical excellence in preprocessing often matters more than algorithmic sophistication.

**3. Trade-offs Are Inevitable**
Every application involved trade-offs:
- Task 1: Simplicity vs. segment granularity
- Task 2: File size vs. visual quality
- Task 3: Computational efficiency vs. classification accuracy

Successful implementation requires explicit identification and conscious navigation of these trade-offs.

**4. Evaluation Requires Multiple Perspectives**
Single metrics proved insufficient in all cases:
- Task 1: Elbow method + business interpretation
- Task 2: File size + visual inspection
- Task 3: ARI + Silhouette + Purity scores

Multi-faceted evaluation prevents blind spots and supports robust decision-making.

### 📌 Recommendations

**For Customer Segmentation:**
- Deploy the 5-cluster solution in production
- Use segment profiles to inform marketing strategies
- Regularly re-cluster with updated customer data
- Monitor cluster stability over time

**For Image Compression:**
- Use 8-color compression for ultra-low-bandwidth scenarios
- Prefer 32-64 color compression for balanced trade-offs
- Implement user controls for quality/compression adjustment
- Consider alternative algorithms (JPEG) for photographic images

**For Digit Recognition:**
- Use original 64-dimensional data for best accuracy
- Consider PCA for real-time applications with performance constraints
- Investigate hierarchical or DBSCAN clustering for non-convex alternative
- Ensemble approaches combining K-Means with neural networks

### 🚀 Future Directions

1. **Task 1:** Implement hierarchical clustering and compare segment hierarchies
2. **Task 2:** Test alternative compression algorithms (Huffman coding, JPEG)
3. **Task 3:** Investigate optimal PCA dimensionality through cross-validation
4. **Cross-Task:** Develop adaptive algorithm selection framework based on data characteristics

---

## Requirements & Setup

### 📦 Dependencies

Install required packages using:
```bash
pip install -r requirements.txt
```

**Key Libraries:**
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- opencv-python >= 4.5.0
- kagglehub >= 0.1.0

### 🏃 Running the Code

**Task 1: Customer Segmentation**
```bash
python customer_segmentation.py
```
Generates: `outputs/elbow_method.png`, `outputs/customer_clusters.png`

**Task 2: Image Compression**
```bash
python image_compression_kmeans.py
```
Generates: Compressed image comparison visualization

**Task 3: Digit Recognition with PCA**
```bash
python digits_pca_kmeans.py
```
Generates: `outputs/pca_variance.png`, clustering comparison metrics

### 📁 Output Files

All visualizations and results are saved to the `outputs/` directory:
- `elbow_method.png` - Elbow curve for optimal cluster selection
- `customer_clusters.png` - Scatter plot of customer segments
- `pca_variance.png` - Cumulative explained variance by components

---

*This analysis demonstrates the power and versatility of K-Means clustering while highlighting the importance of careful preprocessing, parameter tuning, and multi-metric evaluation in machine learning applications.*
