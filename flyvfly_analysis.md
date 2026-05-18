# Fly-v-Fly F1-Score Analysis

## Issue
The F1-score for the fly-v-fly reproduction is 0.311287, which appears low. The best program shows severe class imbalance in predictions.

## Best Program Analysis

**Program:**
```
output(
  convolve_5_len300(
    map(
      mul(
        affine_positional(),
        add(
          mul(affine_wing(), affine_wing()),
          affine_angular()
        )
      )
    )
  )
)
```

**Metrics:**
- F1-score: 0.311287
- Hamming accuracy: 0.434286
- Per-class F1s: [0.0, 0.654, 0.139, 0.0, 0.066, 0.0, 0.0]

## Key Findings

### 1. Severe Class Imbalance in Predictions

**True Distribution (1050 test samples):**
- Class 0: 363 samples (34.6%)
- Class 1: 480 samples (45.7%)
- Class 2: 66 samples (6.3%)
- Class 3: 21 samples (2.0%)
- Class 4: 61 samples (5.8%)
- Class 5: 41 samples (3.9%)
- Class 6: 18 samples (1.7%)

**Predicted Distribution:**
- Class 0: 0 samples (0%)
- Class 1: 888 samples (84.6%) ⚠️
- Class 2: 6 samples (0.6%)
- Class 3: 2 samples (0.2%)
- Class 4: 61 samples (5.8%)
- Class 5: 0 samples (0%)
- Class 6: 93 samples (8.9%)

The model is heavily biased toward predicting class 1, which accounts for 84.6% of all predictions despite only being 45.7% of the true labels.

### 2. Potential Issues

#### A. Feature Selection Differences

**NEAR Original Implementation (`library_functions.py:270-274`):**
```python
def execute_on_batch(self, batch, batch_lens=None):
    features = torch.index_select(batch, 1, self.feature_tensor)
    remaining_features = batch[:,self.full_feature_dim:]
    return self.linear_layer(torch.cat([features, remaining_features], dim=-1))
```

**neurosym-lib Implementation:**
```python
lambda lin, feature_indices=feature_indices: lambda x: lin(x[..., feature_indices])
```

**Difference:** The NEAR implementation concatenates selected features with "remaining features" after `full_feature_dim`. However, since `FLYVFLY_FULL_FEATURE_DIM = 53` and the data has exactly 53 features, `remaining_features` would be empty, so this shouldn't cause a difference.

#### B. Convolution Implementation

The Conv1d operation uses an unusual configuration:
```python
torch.nn.Conv1d(seq_len, 1, kernel_size, padding=..., bias=False)
```

With input shape `(batch, seq_len=300, hidden_dim=16)`, this:
1. Treats time steps as input channels: `(batch, in_channels=300, length=16)`
2. Learns a weighted combination of the 300 time steps for each hidden dimension
3. Outputs `(batch, 1, 16)` which becomes `(batch, 16)` after squeeze

This is **not** a traditional temporal convolution but rather a learnable temporal aggregation. While unusual, this may be intentional.

#### C. Structural Cost Penalty

Command used: `--structural-cost-penalty 0.1`

The structural cost penalty of 0.1 is **100x higher** than the default value of 0.001. This may be:
- Preventing more complex programs from being explored
- Biasing search toward simpler programs that may not fit the data well

#### D. Training Configuration

From `benchmark_flyvfly.py`:
- Batch size: 2048 (large)
- Learning rate: 1e-4 (from default, line 336)
- Neural epochs: 30 (search)
- Final epochs: 40 (best program)

Potential issues:
- Large batch size may lead to poor generalization
- Fixed LR without scheduling may prevent fine-tuning
- Class imbalance in training data not addressed

### 3. Missing Baseline Comparison

The NEAR repository (`/home/asehgal/NEAR/near/near_code/`) does not contain:
- Fly-v-fly experiment results
- Fly-v-fly command line examples
- Expected performance metrics

The `report.md` only discusses CRIM-13 results, not fly-v-fly.

## Recommendations

### 1. Verify Structural Cost Penalty
Try with the original NEAR default:
```bash
--structural-cost-penalty 0.001
```

### 2. Check for Class Weighting
The NEAR original code supports class weights (`train.py:96-97`):
```python
parser.add_argument('--class_weights', type=str, required=False, default=None,
                    help="weights for each class in the loss function")
```

With the severe class imbalance (class 1 is 45.7%, class 6 is 1.7%), class weighting may be essential.

### 3. Investigate Loss Function
The current implementation uses `ce_loss` with no class weighting. Check if this matches the NEAR paper's approach for imbalanced classification.

### 4. Verify DSL Matches Paper
The NEAR paper may describe the exact DSL configuration used for fly-v-fly. Check if:
- The affine selectors match
- The convolution approach is correct
- Any productions are missing

### 5. Check Data Preprocessing
Verify that the data loading and preprocessing matches the original NEAR implementation:
- Normalization
- Feature scaling
- Sequence padding/truncation

## Next Steps

1. **Search for NEAR paper's reported fly-v-fly results** to establish a baseline
2. **Run with structural cost penalty = 0.001** (1000x smaller)
3. **Add class weighting** to the loss function
4. **Compare full DSL** between neurosym-lib and NEAR implementations
5. **Check if convolution implementation** is intended or needs fixing
