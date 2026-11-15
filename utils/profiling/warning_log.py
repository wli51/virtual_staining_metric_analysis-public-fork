"""Route CP-call RuntimeWarnings to JSONL without hiding exceptions or writer warnings."""
import json
import uuid
import warnings
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def logged_cp_batches(batches, path, context):
    """Wrap sequential CP iteration only; restore warning handlers before each yield.

    Each record aggregates identical emitted warnings within one next() call.
    Counts are warning emissions, not counts of affected cells or NaN features.
    Exceptions propagate, including after warnings were logged. Other warning
    categories retain their usual console destination. This wrapper assumes the
    existing single-threaded profiler runner; do not use for concurrent threads.
    """
    path = Path(path)
    iterator = iter(batches)
    attempt_id = uuid.uuid4().hex
    while True:
        captured = []
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always', RuntimeWarning)
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
        finally:
            counts = Counter()
            for warning in captured:
                if issubclass(warning.category, RuntimeWarning):
                    counts[(warning.category.__name__, str(warning.message),
                            warning.filename, warning.lineno)] += 1
                else:
                    warnings.showwarning(warning.message, warning.category,
                                         warning.filename, warning.lineno)
            if counts:
                with path.open('a', encoding='utf-8') as stream:
                    for (category, message, filename, lineno), count in counts.items():
                        event = {**context, 'attempt_id': attempt_id,
                                 'timestamp_utc': datetime.now(UTC).isoformat(),
                                 'category': category, 'message': message,
                                 'filename': filename, 'lineno': lineno, 'count': count}
                        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        yield batch
