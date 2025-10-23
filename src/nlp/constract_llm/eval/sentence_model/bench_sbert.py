import logging
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from nlp.common.utils.file.json import save_as_indented_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_device_index(device: str) -> int:
    """Extract CUDA device index from a device string like 'cuda:1'. Defaults to 0.
    If CPU, returns 0.
    """
    if device.startswith('cuda') and ':' in device:
        try:
            return int(device.split(':')[1])
        except Exception:
            return 0
    return 0


def _resolve_dtype(dtype_str: str | None) -> torch.dtype | str | None:
    if dtype_str is None:
        return None
    m = dtype_str.lower()
    match m:
        case 'float16' | 'fp16' | 'half':
            return torch.float16
        case 'bfloat16' | 'bf16':
            return torch.bfloat16
        case 'float32' | 'fp32' | 'float':
            return torch.float32
        case 'auto':
            return 'auto'
        case _:
            raise ValueError(f'Unsupported dtype: {dtype_str}')


class GPUMonitor:
    """Background GPU monitor.

    - Tries `pynvml` first.
    - Falls back to calling `nvidia-smi` if available.
    - If neither is available, only records PyTorch peak memory (set externally).
    """

    def __init__(self, device_index: int = 0, interval_sec: float = 0.1) -> None:
        self.device_index = device_index
        self.interval_sec = interval_sec
        self.utilization: list[float] = []
        self.mem_used_mib: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._backend: str | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True

        def _run() -> None:
            backend_selected = False
            pynvml = None
            try:
                import pynvml as _p

                pynvml = _p
                _p.nvmlInit()
                _ = _p.nvmlDeviceGetHandleByIndex(self.device_index)
                self._backend = 'pynvml'
                backend_selected = True
                logger.debug('GPUMonitor: using pynvml backend')
            except Exception:
                pass

            import shutil
            import subprocess

            smi_path = shutil.which('nvidia-smi')
            if not backend_selected and smi_path:
                self._backend = 'nvidia-smi'
                backend_selected = True
                logger.debug('GPUMonitor: using nvidia-smi backend')

            while self._running:
                try:
                    if self._backend == 'pynvml' and pynvml is not None:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        self.utilization.append(float(util.gpu))
                        self.mem_used_mib.append(float(mem.used) / (1024**2))
                    elif self._backend == 'nvidia-smi' and smi_path is not None:
                        cmd = [
                            smi_path,
                            f'--id={self.device_index}',
                            '--query-gpu=utilization.gpu,memory.used',
                            '--format=csv,noheader,nounits',
                        ]
                        import subprocess

                        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
                        if out:
                            util_s, mem_s = out.split(',')
                            self.utilization.append(float(util_s.strip()))
                            self.mem_used_mib.append(float(mem_s.strip()))
                except Exception:
                    pass
                time.sleep(self.interval_sec)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)

    def max_utilization(self) -> float | None:
        return max(self.utilization) if self.utilization else None

    def max_memory_used_mib(self) -> float | None:
        return max(self.mem_used_mib) if self.mem_used_mib else None


def _extract_text(example: dict[str, Any], text_fields: list[str]) -> str:
    parts: list[str] = []
    for key in text_fields:
        val = example.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, (list, tuple)):
            parts.extend([str(x) for x in val if isinstance(x, (str, int, float))])
        else:
            parts.append(str(val))
    return ' '.join(parts).strip()


def _load_texts_from_hf(
    dataset_name_or_path: str,
    split: str,
    text_fields: list[str],
    *,
    dataset_config_name: str | None = None,
    dataset_revision: str | None = None,
    streaming: bool = True,
    max_use_samples: int | None = None,
) -> list[str]:
    """Load texts from Hugging Face Datasets.

    - Concatenate values of `text_fields` with a space to form a single text.
    - When `streaming=True`, iterate lazily and collect up to `max_use_samples`.
    """
    load_kwargs: dict[str, Any] = {
        'split': split,
        'streaming': streaming,
    }
    if dataset_revision is not None:
        load_kwargs['revision'] = dataset_revision

    # name -> config name
    if dataset_config_name is not None:
        ds = load_dataset(dataset_name_or_path, dataset_config_name, **load_kwargs)
    else:
        ds = load_dataset(dataset_name_or_path, **load_kwargs)

    texts: list[str] = []
    if streaming:
        for ex in ds:  # type: ignore[assignment]
            txt = _extract_text(ex, text_fields)
            if txt:
                texts.append(txt)
            if max_use_samples is not None and len(texts) >= max_use_samples:
                break
    else:
        # Materialize then slice
        for i, ex in enumerate(ds):  # type: ignore[assignment]
            if max_use_samples is not None and i >= max_use_samples:
                break
            txt = _extract_text(ex, text_fields)
            if txt:
                texts.append(txt)

    if not texts:
        raise ValueError('No texts loaded from the dataset. Check text_fields and split.')
    return texts


class BenchmarkRecord(BaseModel):
    seq_len: int = Field(...)
    batch_size: int = Field(...)
    total_time_sec: float = Field(...)
    avg_time_per_text_ms: float = Field(...)
    throughput_texts_per_sec: float = Field(...)
    max_gpu_util_percent: float | None = Field(None)
    peak_vram_mib: float | None = Field(None)
    torch_peak_allocated_mib: float | None = Field(None)
    error: str | None = Field(None)


def benchmark_embeddings(
    model_name_or_path: str,
    *,
    dataset_name_or_path: str,
    dataset_split: str,
    text_fields: list[str],
    batch_sizes: Iterable[int],
    seq_lengths: Iterable[int],
    output_dir: str | Path,
    device: str | None = None,
    dtype: str | None = None,
    normalize_embeddings: bool = True,
    max_use_samples: int | None = None,
    monitor_interval_ms: int = 100,
    warmup_fraction: float = 0.0,
    dataset_config_name: str | None = None,
    dataset_revision: str | None = None,
    streaming: bool = True,
    show_progress: bool = True,
    # Model init options similar to run_st.py
    cache_dir: str | None = None,
    model_revision: str | None = None,
    token: str | None = None,
    trust_remote_code: bool | None = None,
    attn_implementation: str | None = None,
    low_cpu_mem_usage: bool | None = None,
    use_fast_tokenizer: bool | None = None,
    # Extra passthrough kwargs
    model_extra_kwargs: dict[str, Any] | None = None,
    tokenizer_extra_kwargs: dict[str, Any] | None = None,
    common_extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Sweep batch sizes and sequence lengths to measure throughput and GPU usage for Sentence-Transformers.

    Notes
    -----
    - Update `model.max_seq_length` per combination and measure performance.
    - Sample GPU utilization via `pynvml` or `nvidia-smi` when available.
    - Reference: UKPLab/sentence-transformers #2551.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = _load_texts_from_hf(
        dataset_name_or_path=dataset_name_or_path,
        split=dataset_split,
        text_fields=text_fields,
        dataset_config_name=dataset_config_name,
        dataset_revision=dataset_revision,
        streaming=streaming,
        max_use_samples=max_use_samples,
    )
    n_texts = len(texts)
    logger.info(f'Loaded {n_texts} texts from HF dataset: {dataset_name_or_path} [split={dataset_split}]')

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        logger.warning('CUDA is not available. Falling back to CPU.')
        device = 'cpu'

    torch_dtype = _resolve_dtype(dtype)

    def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if v is not None}

    common_kwargs = _drop_none(
        {
            'cache_folder': cache_dir,
            'revision': model_revision,
            'token': token,
            'trust_remote_code': trust_remote_code,
        }
    )
    if common_extra_kwargs:
        common_kwargs.update(_drop_none(common_extra_kwargs))

    model_kwargs = _drop_none(
        {
            'torch_dtype': torch_dtype,
            'attn_implementation': attn_implementation,
            'low_cpu_mem_usage': low_cpu_mem_usage,
        }
    )
    if model_extra_kwargs:
        model_kwargs.update(_drop_none(model_extra_kwargs))

    tokenizer_kwargs = _drop_none({'use_fast': use_fast_tokenizer} if use_fast_tokenizer is not None else {})
    if tokenizer_extra_kwargs:
        tokenizer_kwargs.update(_drop_none(tokenizer_extra_kwargs))

    logger.info(
        f'Loading model: {model_name_or_path} (device={device}, dtype={torch_dtype}, '
        f'model_kwargs={model_kwargs}, tokenizer_kwargs={tokenizer_kwargs}, common={common_kwargs})'
    )
    model = SentenceTransformer(
        model_name_or_path,
        device=device,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        **common_kwargs,
    )

    if warmup_fraction and warmup_fraction > 0:
        warmup_n = max(1, int(n_texts * min(1.0, max(0.0, warmup_fraction))))
        logger.info(f'Warmup encode for {warmup_n} samples...')
        _ = model.encode(
            texts[:warmup_n],
            batch_size=min(4, warmup_n),
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress,
        )

    monitor_interval_sec = max(1, monitor_interval_ms) / 1000.0
    device_index = _parse_device_index(device)

    records: list[BenchmarkRecord] = []

    bs_list = [int(b) for b in batch_sizes]
    sl_list = [int(s) for s in seq_lengths]
    combo_total = len(bs_list) * len(sl_list)
    combo_pbar = tqdm(total=combo_total, disable=not show_progress, desc='Combinations', dynamic_ncols=True)

    for seq_len in sl_list:
        # Apply seq length
        try:
            model.max_seq_length = int(seq_len)
        except Exception as e:  # pragma: no cover
            logger.warning(f'Failed to set model.max_seq_length={seq_len}: {e}')

        for bs in bs_list:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            monitor = GPUMonitor(device_index=device_index, interval_sec=monitor_interval_sec)
            monitor.start()

            start = time.perf_counter()
            error_msg: str | None = None
            try:
                with torch.inference_mode():
                    _ = model.encode(
                        texts,
                        batch_size=int(bs),
                        normalize_embeddings=normalize_embeddings,
                        show_progress_bar=show_progress,
                        convert_to_numpy=True,
                    )
            except RuntimeError as e:
                error_msg = str(e)
                logger.error(f'Error at seq_len={seq_len}, batch_size={bs}: {e}')
            finally:
                end = time.perf_counter()
                monitor.stop()

            combo_pbar.update(1)
            combo_pbar.set_postfix({'seq_len': seq_len, 'batch_size': bs})

            elapsed = end - start

            max_util = monitor.max_utilization()
            max_mem = monitor.max_memory_used_mib()
            torch_peak_allocated_mib: float | None = None
            if torch.cuda.is_available():
                try:
                    torch_peak_allocated_mib = float(torch.cuda.max_memory_allocated()) / (1024**2)
                except Exception:
                    pass

            if error_msg is None and elapsed > 0:
                avg_ms = (elapsed / n_texts) * 1000.0
                tps = n_texts / elapsed
            else:
                avg_ms = float('nan')
                tps = float('nan')

            records.append(
                BenchmarkRecord(
                    seq_len=int(seq_len),
                    batch_size=int(bs),
                    total_time_sec=float(elapsed),
                    avg_time_per_text_ms=float(avg_ms),
                    throughput_texts_per_sec=float(tps),
                    max_gpu_util_percent=max_util if max_util is not None else None,
                    peak_vram_mib=max_mem if max_mem is not None else None,
                    torch_peak_allocated_mib=torch_peak_allocated_mib,
                    error=error_msg,
                )
            )

    combo_pbar.close()

    results_dict: dict[str, Any] = {
        'model_name_or_path': model_name_or_path,
        'dataset': {
            'name_or_path': dataset_name_or_path,
            'split': dataset_split,
            'config_name': dataset_config_name,
            'revision': dataset_revision,
            'text_fields': text_fields,
            'streaming': streaming,
        },
        'device': device,
        'dtype': dtype,
        'normalize_embeddings': normalize_embeddings,
        'n_texts': n_texts,
        'batch_sizes': list(map(int, batch_sizes)),
        'seq_lengths': list(map(int, seq_lengths)),
        'records': [r.model_dump() for r in records],
        'meta': {
            'timestamp': int(time.time()),
            'source': 'bench_sbert',
            'note': 'Batch size / seq length sweep for Sentence-Transformers.',
        },
    }

    json_path = out_dir / 'benchmark_results.json'
    save_as_indented_json(results_dict, json_path)
    logger.info(f'Saved results: {json_path}')

    df = pd.DataFrame(results_dict['records'])
    csv_path = out_dir / 'benchmark_results.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f'Saved CSV: {csv_path}')

    try:
        import japanize_matplotlib  # noqa: F401
    except Exception:
        pass

    def _pivot_and_plot(
        value_col: str,
        title: str,
        cmap: str,
        filename: str,
        fmt: str = '.1f',
    ) -> None:
        valid = df.copy()
        valid = valid[valid['error'].isna()]
        if valid.empty:
            logger.warning(f'No valid records to plot for {value_col}')
            return
        pv = valid.pivot_table(index='seq_len', columns='batch_size', values=value_col, aggfunc='median')
        plt.figure(figsize=(10, 6))
        sns.heatmap(pv, annot=True, fmt=fmt, cmap=cmap)
        plt.title(title)
        plt.ylabel('model_seq_len')
        plt.xlabel('batch_size')
        plt.tight_layout()
        out_path = out_dir / filename
        plt.savefig(out_path)
        plt.close()
        logger.info(f'Saved plot: {out_path}')

    _pivot_and_plot(
        'avg_time_per_text_ms',
        'Average inference time per text [ms]',
        'mako_r',
        'avg_time_per_text_ms.png',
        fmt='.1f',
    )
    _pivot_and_plot(
        'throughput_texts_per_sec',
        'Throughput [texts/sec]',
        'YlGnBu',
        'throughput_texts_per_sec.png',
        fmt='.1f',
    )
    # GPU util and memory (if available)
    if df['max_gpu_util_percent'].notna().any():
        _pivot_and_plot(
            'max_gpu_util_percent',
            'Max GPU utilization [%]',
            'OrRd',
            'max_gpu_util_percent.png',
            fmt='.0f',
        )
    if df['peak_vram_mib'].notna().any():
        _pivot_and_plot(
            'peak_vram_mib',
            'Max GPU memory usage [MiB] (nvidia-smi)',
            'BuPu',
            'peak_vram_mib.png',
            fmt='.0f',
        )
    if df['torch_peak_allocated_mib'].notna().any():
        _pivot_and_plot(
            'torch_peak_allocated_mib',
            'PyTorch peak allocated memory [MiB]',
            'PuRd',
            'torch_peak_allocated_mib.png',
            fmt='.0f',
        )

    return results_dict
