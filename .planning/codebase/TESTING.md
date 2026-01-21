# Testing Patterns

**Analysis Date:** 2026-01-21

## Test Framework

**Status:** No formal test framework configured

**Framework Details:**
- No pytest, unittest, or other testing framework detected in requirements.txt
- No test runner configuration files (pytest.ini, setup.cfg, tox.ini)
- No test directory structure (tests/, test/)
- Testing relies on manual `test_*()` functions and `if __name__ == "__main__"` blocks

**Assertion Library:**
- Built-in `assert` statements from Python standard library
- No third-party assertion libraries (pytest, nose, unittest)

**Run Commands:**
```bash
# Run individual module tests
python src/data/dataset.py
python src/models/encoder.py
python src/evaluation/evaluator.py
python src/data/preprocessing.py
python src/models/blocks.py

# Run specific test function (if isolated)
python -m pytest src/data/dataset.py::test_dataset  # Not implemented yet
```

## Test File Organization

**Location:** Tests are co-located with source code

**Pattern:**
- Test functions in same file as implementation
- Module-level `test_*()` functions at end of each file
- Main guard wraps test execution: `if __name__ == "__main__": test_function()`

**Examples:**
- `src/data/dataset.py` contains `test_dataset()` function
- `src/data/preprocessing.py` contains `test_preprocessing()` function
- `src/evaluation/evaluator.py` contains `test_evaluator()` function
- `src/models/encoder.py` contains `test_encoder()` function
- `src/models/blocks.py` contains `test_blocks()` function

**Naming:**
- Test functions: `test_<component>()`
- Main entry point: `if __name__ == "__main__": test_<component>()`

## Test Structure

**Suite Organization Pattern:**

From `src/data/dataset.py`:
```python
def test_dataset():
    """Test dataset classes."""
    print("Testing SARPatchDataset...")

    # Create synthetic patches
    np.random.seed(42)
    test_patches = np.random.rand(100, 256, 256).astype(np.float32)

    # Test without augmentation
    dataset = SARPatchDataset(test_patches, augment=False)
    sample = dataset[0]

    assert sample.shape == (1, 256, 256), f"Wrong shape: {sample.shape}"
    print(f"✓ Shape correct: {sample.shape}")

    assert sample.dtype == torch.float32, f"Wrong dtype: {sample.dtype}"
    print(f"✓ Dtype correct: {sample.dtype}")

    # Test augmentation creates variety
    dataset_aug = SARPatchDataset(test_patches, augment=True)
    sample1 = dataset_aug[0]
    sample2 = dataset_aug[0]  # Same index

    if not torch.allclose(sample1, sample2):
        print("✓ Augmentation creates variety")
    else:
        print("⚠ Warning: Augmentation may not be working")

    print("All dataset tests passed!")
```

**Test Structure Elements:**
1. Setup: Create test data (fixtures inline)
2. Test execution: Invoke code under test
3. Assertions: Validate results with `assert`
4. Output: Print status with `print()` and visual indicators (`✓`, `⚠`)
5. Summary: Print completion message

**Patterns Observed:**

From `src/evaluation/evaluator.py`:
```python
def test_evaluator():
    """Test evaluation framework."""
    print("=" * 60)
    print("EVALUATOR TEST")
    print("=" * 60)

    # Create dummy model and data
    from torch.utils.data import TensorDataset, DataLoader

    # Simple passthrough "model" for testing
    class DummyModel(torch.nn.Module):
        def __init__(self, noise_level=0.05):
            super().__init__()
            self.noise_level = noise_level
            self.encoder = torch.nn.Conv2d(1, 64, 1)

        def forward(self, x):
            z = self.encoder(x)
            x_hat = x + self.noise_level * torch.randn_like(x)
            x_hat = x_hat.clamp(0, 1)
            return x_hat, z

        def encode(self, x):
            return self.encoder(x)

    model = DummyModel(noise_level=0.05)

    # Create test data
    test_data = torch.rand(20, 1, 64, 64)
    dataset = TensorDataset(test_data)
    dataloader = DataLoader(dataset, batch_size=4)

    # ... rest of test
```

**Key Patterns:**
- Dummy/mock objects created inline as needed: `DummyModel`, `SimpleLoader`
- Test data generated synthetically: `torch.rand()`, `np.random.rand()`
- Random seed fixed: `np.random.seed(42)` for reproducibility
- Wrapper classes for test compatibility: `SimpleLoader` wraps DataLoader

## Mocking

**Framework:** No external mocking library (unittest.mock not used)

**Patterns:**
- Inline mock classes: `DummyModel`, `SimpleLoader`
- Mock implements minimal interface needed for test
- Pass-through or stub implementations

**Example from `src/evaluation/evaluator.py`:**
```python
class DummyModel(torch.nn.Module):
    def __init__(self, noise_level=0.05):
        super().__init__()
        self.noise_level = noise_level
        self.encoder = torch.nn.Conv2d(1, 64, 1)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = x + self.noise_level * torch.randn_like(x)  # Add noise
        x_hat = x_hat.clamp(0, 1)
        return x_hat, z

    def encode(self, x):
        return self.encoder(x)
```

**What to Mock:**
- External dependencies with complex state: models, data loaders
- Keep mocks simple with predictable behavior

**What NOT to Mock:**
- Pure utility functions: use real implementations
- Metric computations: validate actual calculations
- Data transformations: test real transforms

## Fixtures and Factories

**Test Data:**
- Generated synthetically in each test function
- Random seeds for reproducibility: `np.random.seed(42)`
- Use domain-appropriate distributions: exponential for SAR intensity

**Example from `src/data/preprocessing.py`:**
```python
def test_preprocessing():
    """Test preprocessing functions."""
    print("Testing preprocessing...")

    # Create synthetic SAR-like data
    np.random.seed(42)
    # Exponential distribution mimics SAR intensity
    test_image = np.random.exponential(0.1, (512, 512)).astype(np.float32)

    # Test preprocessing
    normalized, params = preprocess_sar_complete(test_image)

    assert normalized.min() >= 0, "Normalized min should be >= 0"
    assert normalized.max() <= 1, "Normalized max should be <= 1"
    print(f"✓ Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
```

**Location:** Fixtures generated inline in test functions (no separate fixture files)

**Pattern:**
- Create minimal test data needed for test
- Use realistic distributions for domain data
- Store in local variables within test scope

## Coverage

**Requirements:** Not explicitly defined - no coverage.py config detected

**View Coverage:**
```bash
# Not currently configured, would need:
pip install coverage
coverage run -m pytest
coverage report
coverage html
```

## Test Types

**Unit Tests:**
- Scope: Individual functions and classes
- Approach: Create input → call function → assert output
- Examples: `test_dataset()` validates SARPatchDataset tensor conversion and augmentation
- Granularity: Test one behavior per assertion

**Integration Tests:**
- Scope: Multi-component workflows
- Approach: Set up data pipeline → run evaluator → check metrics
- Examples: `test_evaluator()` tests evaluator with dummy model and dataloader
- Granularity: Tests pipeline end-to-end

**E2E Tests:**
- Framework: Not used
- Could be added via: pytest with full system tests
- Would require: integration test files in dedicated directory

## Common Patterns

**Synthetic Data Creation:**

From `src/models/encoder.py`:
```python
def test_encoder():
    """Test encoder implementation."""
    print("Testing SAREncoder...")

    encoder = SAREncoder(latent_channels=64)
    x = torch.randn(2, 1, 256, 256)
    z = encoder(x)

    assert z.shape == (2, 64, 16, 16), f"Wrong shape: {z.shape}"
    print(f"✓ Output shape correct: {z.shape}")

    # Test gradient flow
    loss = z.mean()
    loss.backward()
    print("✓ Gradients flow correctly")

    # Parameter count
    params = sum(p.numel() for p in encoder.parameters())
    print(f"✓ Parameters: {params:,}")

    print("All encoder tests passed!")
```

**Gradient Testing:**
- Validates backprop works: `loss.backward()`
- Checks gradients exist: implicit check when backward succeeds

**Shape Assertions:**
- Standard pattern: `assert tensor.shape == expected, f"Wrong shape: {tensor.shape}"`
- Used to validate layer outputs match architecture

**Dtype Assertions:**
- Ensures correct precision: `assert sample.dtype == torch.float32`
- Critical for GPU/numerical correctness

**Roundtrip Testing:**

From `src/data/preprocessing.py`:
```python
# Test inverse (for non-clipped values)
image_db = 10 * np.log10(np.maximum(test_image, 1e-10))
non_clipped = (image_db >= params['vmin']) & (image_db <= params['vmax'])

error = np.abs(test_image[non_clipped] - reconstructed[non_clipped])
relative_error = error / test_image[non_clipped]

print(f"✓ Roundtrip mean relative error: {relative_error.mean()*100:.4f}%")
```

**Range/Bounds Checking:**
- Validates normalized output: `assert normalized.min() >= 0 and normalized.max() <= 1`
- Common for preprocessing pipelines

## Known Test Gaps

**Areas Without Tests:**
- `src/losses/` - No test functions in combined.py, mse.py, ssim.py
- `src/inference/compressor.py` - No test_inference()
- `src/compression/histogram.py` - No test functions
- `src/training/` - No training tests (trainer.py not yet implemented)
- Scripts (scripts/train.py, scripts/evaluate.py) - No test functions

**Why Tests Are Incomplete:**
- Code marked with `raise NotImplementedError()` - can't test stubs
- Files with TODO comments - implementation deferred
- Modules still in development stage

## Recommendations for Testing

**To Implement Full Coverage:**
1. Set up pytest: `pip install pytest pytest-cov`
2. Create test directory: `tests/` parallel to `src/`
3. Add pytest.ini configuration
4. Implement test functions for all public APIs
5. Add CI/CD pipeline to run tests on commits
6. Set coverage targets (e.g., 80% minimum)

**File Structure Template:**
```
tests/
├── conftest.py              # pytest fixtures
├── test_data_dataset.py     # Unit tests for src/data/dataset.py
├── test_data_preprocessing.py
├── test_models_encoder.py
├── test_models_decoder.py
├── test_models_autoencoder.py
├── test_evaluation_evaluator.py
├── test_losses.py
├── integration/
│   └── test_training_pipeline.py
└── e2e/
    └── test_full_workflow.py
```

---

*Testing analysis: 2026-01-21*
