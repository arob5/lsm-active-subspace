"""Inference methods backed by EnsKit, registered with ProbPipe's ``condition_on``.

Importing this package registers them. Each runs only when a call names it, as in
``condition_on.using(method="enskit_eki")`` or ``problem.condition(method="enskit_eki")``.
"""

from probpipe import inference_method_registry

from lsmas.inference.enskit_eki import EnsKitEKI
from lsmas.inference.step_record import StepCallback, StepRecord, history_dataset

__all__ = ["EnsKitEKI", "StepCallback", "StepRecord", "history_dataset"]

inference_method_registry.register(EnsKitEKI())
