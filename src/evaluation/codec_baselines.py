"""
Traditional Codec Baselines for SAR Image Compression Comparison

This module implements JPEG-2000 and JPEG codecs for baseline comparison
with the autoencoder compression.

Classes:
    Codec: Abstract base class for compression codecs
    JPEG2000Codec: JPEG-2000 (wavelet-based) codec implementation
    JPEGCodec: JPEG (DCT-based) codec implementation
    CodecEvaluator: Evaluation wrapper for codec comparison

Design Decisions:
    - WebP excluded per FR4.11 (JPEG-2000 + JPEG provide sufficient coverage)
    - JPEG-2000: High-quality wavelet-based (best traditional for continuous-tone)
    - JPEG: DCT-based, ubiquitous baseline for comparison
    - Binary search calibration achieves target compression ratios within 20%

References:
    - Phase 3 Plan 02: Codec Baselines
    - FR4.11: "optionally JPEG, WebP" for baselines
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import json

import numpy as np
import cv2
from tqdm import tqdm

from .metrics import SARMetrics


class Codec(ABC):
    """
    Abstract base class for image compression codecs.

    Subclasses must implement encode() and decode() methods.

    Attributes:
        name: Codec identifier (e.g., "jpeg2000", "jpeg")
        min_quality: Minimum quality parameter value
        max_quality: Maximum quality parameter value
        quality_direction: "lower" if lower values = more compression, "higher" otherwise
    """

    name: str = "base"
    min_quality: int = 0
    max_quality: int = 100
    quality_direction: str = "higher"  # "higher" or "lower" quality = less compression

    @abstractmethod
    def encode(self, image: np.ndarray, quality: int) -> bytes:
        """
        Encode image to bytes.

        Args:
            image: Input image, float32 in range [0, 1]
            quality: Codec-specific quality parameter

        Returns:
            Encoded bytes
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, encoded: bytes) -> np.ndarray:
        """
        Decode bytes back to image.

        Args:
            encoded: Encoded byte data

        Returns:
            Decoded image as float32 in range [0, 1]
        """
        raise NotImplementedError

    def get_compression_ratio(self, image: np.ndarray, encoded: bytes) -> float:
        """
        Calculate achieved compression ratio.

        Args:
            image: Original image
            encoded: Encoded bytes

        Returns:
            Compression ratio (original_bytes / compressed_bytes)
        """
        original_bytes = image.size * 4  # float32 = 4 bytes
        return original_bytes / len(encoded)

    def calibrate_quality(self, target_ratio: float, sample_image: np.ndarray,
                          tolerance: float = 0.2) -> int:
        """
        Binary search to find quality parameter achieving target compression ratio.

        Args:
            target_ratio: Target compression ratio (e.g., 16.0 for 16x)
            sample_image: Sample image for calibration
            tolerance: Acceptable deviation from target (0.2 = 20%)

        Returns:
            Quality parameter achieving closest to target ratio
        """
        original_bytes = sample_image.size * 4
        low, high = self.min_quality, self.max_quality
        best_param = (low + high) // 2
        best_diff = float('inf')

        # Binary search: find quality parameter that achieves target compression ratio
        # For JPEG-2000: lower quality value = MORE compression (higher ratio)
        # For JPEG: lower quality value = MORE compression (higher ratio)
        #
        # Both codecs: lower quality -> smaller file -> higher compression ratio
        # So quality and compression ratio have inverse relationship

        while low <= high:
            mid = (low + high) // 2

            try:
                encoded = self.encode(sample_image, mid)
                achieved_ratio = original_bytes / len(encoded)
            except Exception:
                # If encoding fails, adjust search direction
                high = mid - 1
                continue

            diff = abs(achieved_ratio - target_ratio) / target_ratio
            if diff < best_diff:
                best_diff = diff
                best_param = mid

            # Adjust search based on achieved vs target
            # Both codecs: lower quality = more compression = higher ratio
            if achieved_ratio > target_ratio:
                # Too much compression, need higher quality parameter
                low = mid + 1
            else:
                # Not enough compression, need lower quality parameter
                high = mid - 1

            # Early exit if within tolerance
            if diff < tolerance:
                break

        return best_param

    def roundtrip(self, image: np.ndarray, quality: int) -> Tuple[np.ndarray, bytes]:
        """
        Encode and decode an image (roundtrip).

        Args:
            image: Input image
            quality: Quality parameter

        Returns:
            Tuple of (decoded_image, encoded_bytes)
        """
        encoded = self.encode(image, quality)
        decoded = self.decode(encoded)
        return decoded, encoded


class JPEG2000Codec(Codec):
    """
    JPEG-2000 codec using OpenCV's jasper/openjpeg backend.

    JPEG-2000 is a wavelet-based codec that typically outperforms DCT-based JPEG
    for continuous-tone images, making it a good upper bound for traditional codecs.

    Quality parameter: IMWRITE_JPEG2000_COMPRESSION_X1000
        - Range: 1-1000
        - Lower values = more compression
        - 1000 = nearly lossless
    """

    name: str = "jpeg2000"
    min_quality: int = 1
    max_quality: int = 1000
    quality_direction: str = "lower"  # lower value = more compression

    def __init__(self):
        """Initialize and check JPEG-2000 support."""
        # Test if JPEG-2000 is available
        # Use 64x64 minimum size - OpenJPEG has issues with very small images
        test_img = np.zeros((64, 64), dtype=np.uint8)
        try:
            success, _ = cv2.imencode('.jp2', test_img,
                                      [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 500])
            if not success:
                raise RuntimeError("Encoding returned failure")
        except Exception as e:
            raise RuntimeError(
                f"JPEG-2000 not supported in this OpenCV build: {e}. "
                "Install opencv-python with jasper/openjpeg support."
            )

    def encode(self, image: np.ndarray, quality: int) -> bytes:
        """
        Encode image to JPEG-2000.

        Args:
            image: Input image, float32 in [0, 1]
            quality: Compression parameter (1-1000, lower = more compression)

        Returns:
            Encoded bytes
        """
        # Convert float32 [0,1] to uint8 [0,255]
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        # Handle grayscale vs color
        if img_uint8.ndim == 2:
            pass  # Already 2D grayscale
        elif img_uint8.ndim == 3 and img_uint8.shape[2] == 1:
            img_uint8 = img_uint8[:, :, 0]  # Remove channel dim

        encode_params = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, int(quality)]
        success, encoded = cv2.imencode('.jp2', img_uint8, encode_params)

        if not success:
            raise RuntimeError(f"JPEG-2000 encoding failed with quality={quality}")

        return encoded.tobytes()

    def decode(self, encoded: bytes) -> np.ndarray:
        """
        Decode JPEG-2000 bytes to image.

        Args:
            encoded: JPEG-2000 encoded bytes

        Returns:
            Decoded image as float32 in [0, 1]
        """
        arr = np.frombuffer(encoded, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

        if decoded is None:
            raise RuntimeError("JPEG-2000 decoding failed")

        # Convert back to float32 [0, 1]
        return decoded.astype(np.float32) / 255.0


class JPEGCodec(Codec):
    """
    JPEG codec using OpenCV.

    JPEG is a DCT-based codec that's ubiquitous but less optimal for SAR images
    due to blocking artifacts at high compression. Good lower-bound baseline.

    Quality parameter: IMWRITE_JPEG_QUALITY
        - Range: 0-100
        - Higher values = less compression (better quality)
        - 95+ = near lossless
        - <30 = heavy compression with visible artifacts
    """

    name: str = "jpeg"
    min_quality: int = 1
    max_quality: int = 100
    quality_direction: str = "higher"  # higher value = less compression

    def encode(self, image: np.ndarray, quality: int) -> bytes:
        """
        Encode image to JPEG.

        Args:
            image: Input image, float32 in [0, 1]
            quality: Quality parameter (0-100, higher = better quality)

        Returns:
            Encoded bytes
        """
        # Convert float32 [0,1] to uint8 [0,255]
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        # Handle grayscale vs color
        if img_uint8.ndim == 2:
            pass  # Already 2D grayscale
        elif img_uint8.ndim == 3 and img_uint8.shape[2] == 1:
            img_uint8 = img_uint8[:, :, 0]  # Remove channel dim

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        success, encoded = cv2.imencode('.jpg', img_uint8, encode_params)

        if not success:
            raise RuntimeError(f"JPEG encoding failed with quality={quality}")

        return encoded.tobytes()

    def decode(self, encoded: bytes) -> np.ndarray:
        """
        Decode JPEG bytes to image.

        Args:
            encoded: JPEG encoded bytes

        Returns:
            Decoded image as float32 in [0, 1]
        """
        arr = np.frombuffer(encoded, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

        if decoded is None:
            raise RuntimeError("JPEG decoding failed")

        # Convert back to float32 [0, 1]
        return decoded.astype(np.float32) / 255.0


class CodecEvaluator:
    """
    Evaluation wrapper for codec comparison.

    Handles calibration, evaluation, caching, and results serialization.
    Uses the same metrics as autoencoder evaluation for fair comparison.

    Example:
        >>> codec = JPEG2000Codec()
        >>> evaluator = CodecEvaluator(codec)
        >>> evaluator.calibrate([8.0, 16.0, 32.0], sample_images)
        >>> result = evaluator.evaluate_single(image, target_ratio=16.0)
        >>> print(f"PSNR: {result['psnr']:.2f}")
    """

    def __init__(self, codec: Codec, cache_dir: Optional[str] = None):
        """
        Initialize evaluator.

        Args:
            codec: Codec instance to evaluate
            cache_dir: Optional directory for caching encoded files
        """
        self.codec = codec
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.calibrated_params: Dict[float, int] = {}  # target_ratio -> quality_param

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, image: np.ndarray, quality: int) -> str:
        """Generate cache key from image hash and quality parameter."""
        img_hash = hashlib.md5(image.tobytes()).hexdigest()[:12]
        return f"{self.codec.name}_{quality}_{img_hash}"

    def _try_cache_load(self, cache_key: str) -> Optional[bytes]:
        """Try to load encoded bytes from cache."""
        if self.cache_dir is None:
            return None
        cache_path = self.cache_dir / f"{cache_key}.bin"
        if cache_path.exists():
            return cache_path.read_bytes()
        return None

    def _cache_save(self, cache_key: str, encoded: bytes) -> None:
        """Save encoded bytes to cache."""
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{cache_key}.bin"
            cache_path.write_bytes(encoded)

    def calibrate(self, target_ratios: List[float], sample_images: List[np.ndarray],
                  tolerance: float = 0.2) -> Dict[float, int]:
        """
        Calibrate quality parameters for target compression ratios.

        Averages calibration across multiple sample images for robustness.

        Args:
            target_ratios: List of target compression ratios (e.g., [8.0, 16.0, 32.0])
            sample_images: List of sample images for calibration
            tolerance: Acceptable deviation from target (0.2 = 20%)

        Returns:
            Dict mapping target_ratio -> calibrated quality parameter
        """
        for target_ratio in target_ratios:
            quality_params = []

            for sample in sample_images:
                param = self.codec.calibrate_quality(target_ratio, sample, tolerance)
                quality_params.append(param)

            # Use median for robustness against outliers
            calibrated_param = int(np.median(quality_params))
            self.calibrated_params[target_ratio] = calibrated_param

        return self.calibrated_params

    def evaluate_single(self, image: np.ndarray, target_ratio: float) -> Dict:
        """
        Encode, decode, and compute metrics for a single image.

        Args:
            image: Input image (float32 in [0, 1])
            target_ratio: Target compression ratio (must be calibrated)

        Returns:
            Dict with metrics and compression info
        """
        # Get calibrated quality parameter
        if target_ratio not in self.calibrated_params:
            # Auto-calibrate if not done
            quality = self.codec.calibrate_quality(target_ratio, image)
            self.calibrated_params[target_ratio] = quality
        else:
            quality = self.calibrated_params[target_ratio]

        # Try cache
        cache_key = self._get_cache_key(image, quality)
        encoded = self._try_cache_load(cache_key)

        if encoded is None:
            encoded = self.codec.encode(image, quality)
            self._cache_save(cache_key, encoded)

        # Decode
        decoded = self.codec.decode(encoded)

        # Compute metrics
        achieved_ratio = self.codec.get_compression_ratio(image, encoded)

        result = {
            'codec': self.codec.name,
            'target_ratio': target_ratio,
            'achieved_ratio': achieved_ratio,
            'quality_param': quality,
            'encoded_bytes': len(encoded),
            'psnr': SARMetrics.psnr(image, decoded),
            'ssim': SARMetrics.ssim(image, decoded),
            'mse': SARMetrics.mse(image, decoded),
            'mae': SARMetrics.mae(image, decoded),
        }

        # Add correlation metrics
        corr = SARMetrics.correlation(image, decoded)
        result['pearson'] = corr['pearson']
        result['spearman'] = corr['spearman']

        return result

    def evaluate_batch(self, images: List[np.ndarray], target_ratio: float,
                       show_progress: bool = True) -> Dict:
        """
        Evaluate across batch of images, return statistics.

        Args:
            images: List of input images
            target_ratio: Target compression ratio
            show_progress: Show tqdm progress bar

        Returns:
            Dict with mean/std/min/max for each metric
        """
        results = []

        iterator = tqdm(images, desc=f"{self.codec.name}@{target_ratio}x") if show_progress else images

        for image in iterator:
            result = self.evaluate_single(image, target_ratio)
            results.append(result)

        # Aggregate statistics
        metrics = ['psnr', 'ssim', 'mse', 'mae', 'achieved_ratio', 'pearson', 'spearman']
        stats = {}

        for metric in metrics:
            values = [r[metric] for r in results]
            stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }

        return {
            'codec': self.codec.name,
            'target_ratio': target_ratio,
            'n_images': len(images),
            'quality_param': self.calibrated_params.get(target_ratio),
            'metrics': stats,
        }

    def evaluate_at_ratios(self, images: List[np.ndarray],
                           target_ratios: List[float],
                           show_progress: bool = True) -> List[Dict]:
        """
        Evaluate at multiple compression ratios.

        Useful for rate-distortion curves.

        Args:
            images: List of input images
            target_ratios: List of target compression ratios
            show_progress: Show progress bar

        Returns:
            List of result dicts, one per ratio
        """
        # Calibrate for all ratios first
        if not all(r in self.calibrated_params for r in target_ratios):
            self.calibrate(target_ratios, images[:min(5, len(images))])

        results = []
        for ratio in target_ratios:
            result = self.evaluate_batch(images, ratio, show_progress)
            results.append(result)

        return results

    def to_json(self, results: Dict, path: str) -> None:
        """
        Save evaluation results to JSON.

        Args:
            results: Results dict from evaluate_batch or evaluate_at_ratios
            path: Output file path
        """
        output = {
            'codec': self.codec.name,
            'evaluation_date': datetime.now().isoformat(),
            'results': results,
        }

        with open(path, 'w') as f:
            json.dump(output, f, indent=2)


def test_codecs():
    """Test codec implementations."""
    print("Testing codec implementations...")

    np.random.seed(42)
    test_img = np.random.rand(256, 256).astype(np.float32)

    # Test JPEG-2000
    print("\n--- JPEG-2000 ---")
    try:
        jp2 = JPEG2000Codec()
        encoded = jp2.encode(test_img, quality=100)
        decoded = jp2.decode(encoded)
        ratio = jp2.get_compression_ratio(test_img, encoded)
        psnr = SARMetrics.psnr(test_img, decoded)
        print(f"  Encoded size: {len(encoded):,} bytes")
        print(f"  Compression ratio: {ratio:.2f}x")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  Shape preserved: {test_img.shape == decoded.shape}")

        # Test calibration
        cal_param = jp2.calibrate_quality(16.0, test_img)
        cal_encoded = jp2.encode(test_img, cal_param)
        cal_ratio = jp2.get_compression_ratio(test_img, cal_encoded)
        print(f"  Calibrated for 16x: param={cal_param}, achieved={cal_ratio:.2f}x")

    except Exception as e:
        print(f"  JPEG-2000 failed: {e}")

    # Test JPEG
    print("\n--- JPEG ---")
    try:
        jpg = JPEGCodec()
        encoded = jpg.encode(test_img, quality=50)
        decoded = jpg.decode(encoded)
        ratio = jpg.get_compression_ratio(test_img, encoded)
        psnr = SARMetrics.psnr(test_img, decoded)
        print(f"  Encoded size: {len(encoded):,} bytes")
        print(f"  Compression ratio: {ratio:.2f}x")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  Shape preserved: {test_img.shape == decoded.shape}")

        # Test calibration
        cal_param = jpg.calibrate_quality(16.0, test_img)
        cal_encoded = jpg.encode(test_img, cal_param)
        cal_ratio = jpg.get_compression_ratio(test_img, cal_encoded)
        print(f"  Calibrated for 16x: param={cal_param}, achieved={cal_ratio:.2f}x")

    except Exception as e:
        print(f"  JPEG failed: {e}")

    print("\n--- CodecEvaluator ---")
    try:
        images = [np.random.rand(256, 256).astype(np.float32) for _ in range(5)]

        for CodecClass, name in [(JPEG2000Codec, 'JPEG-2000'), (JPEGCodec, 'JPEG')]:
            codec = CodecClass()
            evaluator = CodecEvaluator(codec)
            evaluator.calibrate([16.0], images[:2])
            result = evaluator.evaluate_single(images[0], 16.0)
            print(f"  {name}: PSNR={result['psnr']:.2f}, SSIM={result['ssim']:.4f}, "
                  f"ratio={result['achieved_ratio']:.2f}x")
    except Exception as e:
        print(f"  Evaluator test failed: {e}")

    print("\nAll tests complete!")


if __name__ == "__main__":
    test_codecs()
