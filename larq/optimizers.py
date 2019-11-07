"""Neural networks with extremely low-precision weights and activations, such as 
Binarized Neural Networks (BNNs), usually contain a mix of low-precision weights (e.g. 
1-bit) and  higher-precision weights (e.g. 8-bit, 16-bit, or 32-bit). Examples of this 
include the first and last layers of image classificiation models, which have 
higher-precision weights in most BNN architectures from the literature.

Training a BNN, then, consists of optimizing both low-precision and higher-precision
weights. In `larq`, we provide a mechanism to target different bit-precision variables
with different optimizers using the `CaseOptimizer` class. Modeled after the
[`tf.case`](https://www.tensorflow.org/api_docs/python/tf/case) signature,
`CaseOptimizer` accepts pairs of predicates and optimizers. A predicate, given a
variable, decides whether its optimizer should train that variable.

A `CaseOptimizer` behaves much like any other
[Keras optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers), and
once you instantiate it you can pass it to your `model.compile()` as usual. To
instantiate a `CaseOptimzer`, pass one or a list of `(predicate, optimizer)` tuples,
along with a `default` optimizer which trains any variables not claimed by another
optimizer. A variable may not be claimed by more than one optimizer's predicate.

!!! example
    ```python
    case_optimizer = lq.optimizers.CaseOptimizer(
        pred_opt_pairs=(
            lq.optimizers.Bop.is_binary_variable,  # predicate
            lq.optimizers.Bop(threshold=1e-6, gamma=1e-3),  # optimizer
        ),
        default=tf.keras.optimizers.Adam(0.01),
    )
    ```
"""

from larq import utils

if utils.tf_1_14_or_newer():
    from larq.optimizers_v2 import Bop, CaseOptimizer

    __all__ = ["Bop", "CaseOptimizer"]
    assert Bop  # Asserts keep pyflakes happy
    assert CaseOptimizer

else:
    from larq.optimizers_v1 import Bop, XavierLearningRateScaling

    __all__ = ["Bop", "XavierLearningRateScaling"]
    assert Bop
    assert XavierLearningRateScaling
