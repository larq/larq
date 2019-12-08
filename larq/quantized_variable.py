"""Contains QuantizedVariable, a variable that can be quantized in the forward pass."""
import tensorflow as tf
from tensorflow.python.distribute.values import DistributedVariable
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops

from larq import quantized_scope


class QuantizedVariable(tf.Variable):
    """A Variable that can be quantized in the forward pass in applicable contexts."""

    def __init__(self, variable, quantizer=None, precision=None):
        """Creates an QuantizedVariable instance.

        # Arguments
        variable: A floating-point resource variable to wrap.
        quantizer: An optional quantizer to transform the floating-point
            variable to a fake quantized variable.
        precision: An optional integer defining the precision of the quantized
            variable. If `None`, `quantizer.precision` is used.
        """
        if not resource_variable_ops.is_resource_variable(variable):
            raise ValueError(
                f"variable must be of type tf.ResourceVariable, but got: {variable}"
            )
        self.latent_variable = variable
        self.quantizer = quantizer
        self.precision = precision or getattr(quantizer, "precision", None)

    def _quantize(self, value):
        if self.quantizer and quantized_scope.should_quantize():
            return self.quantizer(value)
        return value

    def value(self):
        return self._quantize(self.latent_variable.value())

    def read_value(self):
        return self._quantize(self.latent_variable.read_value())

    def numpy(self):
        return self._quantize(self.latent_variable).numpy()

    def sparse_read(self, *args, **kwargs):
        return self._quantize(self.latent_variable.sparse_read(*args, **kwargs))

    def gather_nd(self, *args, **kwargs):
        return self._quantize(self.latent_variable.gather_nd(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self.latent_variable, name)

    def _dense_var_to_tensor(self, *args, **kwargs):
        return self._quantize(
            self.latent_variable._dense_var_to_tensor(*args, **kwargs)
        )

    def eval(self, session=None):
        return self._quantize(self.latent_variable).eval(session=session)

    def initialized_value(self):
        return self._quantize(self.latent_variable.initialized_value())

    @property
    def initial_value(self):
        return self._quantize(self.latent_variable.initial_value)

    def _should_act_as_resource_variable(self):
        """Pass resource_variable_ops.is_resource_variable check."""
        pass

    def __repr__(self):
        repr_ = (
            f"<{self.__class__.__name__} '{self.name}' "
            f"shape={self.shape} dtype={self.dtype.name}"
        )
        if self.quantizer is not None:
            repr_ += f" quantizer={_get_name(self.quantizer)}"
        if self.precision is not None:
            repr_ += f" precision={self.precision}"
        if tf.executing_eagerly() and not self._in_graph_mode:
            return f"{repr_} numpy={ops.numpy_text(self.read_value(), is_repr=True)}>"
        return f"{repr_}>"

    # Method delegations: We delegate the following methods to self.latent_variable.
    # Each of these methods simply calls the same method on self.latent_variable. The
    # base Variable raises NotImplementedError for most of these, so we must
    # override them.
    #
    # We do not define the following methods from Variable for the following
    # reasons:
    #   * 'experimental_ref': Instead we inherit the definition from Variable.
    #     If we defined and delegated to Variable, the ref of an QuantizedVariable
    #     would be the same as the ref of the underlying variable, which would be
    #     strange as they are different Python objects.

    def set_shape(self, *args, **kwargs):
        return self.latent_variable.set_shape(self, *args, **kwargs)

    @property
    def trainable(self):
        return self.latent_variable.trainable

    @property
    def synchronization(self):
        return self.latent_variable.synchronization

    @property
    def aggregation(self):
        return self.latent_variable.aggregation

    @property
    def constraint(self):
        return self.latent_variable.constraint

    def assign(self, value, use_locking=None, name=None, read_value=True):
        op = self.latent_variable.assign(value, use_locking, name, read_value)
        return _maybe_wrap(op, self.quantizer, self.precision, wrap=read_value)

    def assign_add(self, delta, use_locking=None, name=None, read_value=True):
        op = self.latent_variable.assign_add(delta, use_locking, name, read_value)
        return _maybe_wrap(op, self.quantizer, self.precision, wrap=read_value)

    def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
        op = self.latent_variable.assign_sub(delta, use_locking, name, read_value)
        return _maybe_wrap(op, self.quantizer, self.precision, wrap=read_value)

    def scatter_sub(self, *args, **kwargs):
        var = self.latent_variable.scatter_sub(*args, **kwargs)
        return _maybe_wrap(var, self.quantizer, self.precision)

    def scatter_add(self, *args, **kwargs):
        var = self.latent_variable.scatter_add(*args, **kwargs)
        return _maybe_wrap(var, self.quantizer, self.precision)

    def scatter_max(self, *args, **kwargs):
        var = self.latent_variable.scatter_max(*args, **kwargs)
        return _maybe_wrap(var, self.quantizer, self.precision)

    def scatter_min(self, *args, **kwargs):
        var = self.latent_variable.scatter_min(*args, **kwargs)
        return _maybe_wrap(var, self.quantizer, self.precision)

    def scatter_mul(self, *args, **kwargs):
        var = self.latent_variable.scatter_mul(*args, **kwargs)
        return _maybe_wrap(var, self.quantizer, self.precision)

    def scatter_div(self, *args, **kwargs):
        var = self.latent_variable.scatter_div(*args, **kwargs)
        return _maybe_wrap(var, self.quantizer, self.precision)

    def scatter_update(self, *args, **kwargs):
        var = self.latent_variable.scatter_update(*args, **kwargs)
        return _maybe_wrap(var, self.quantizer, self.precision)

    def batch_scatter_update(self, *args, **kwargs):
        var = self.latent_variable.batch_scatter_update(*args, **kwargs)
        return _maybe_wrap(var, self.quantizer, self.precision)

    def scatter_nd_sub(self, *args, **kwargs):
        var = self.latent_variable.scatter_nd_sub(*args, **kwargs)
        return _maybe_wrap(var, self.quantizer, self.precision)

    def scatter_nd_add(self, *args, **kwargs):
        var = self.latent_variable.scatter_nd_add(*args, **kwargs)
        return _maybe_wrap(var, self.quantizer, self.precision)

    def scatter_nd_update(self, *args, **kwargs):
        var = self.latent_variable.scatter_nd_update(*args, **kwargs)
        return _maybe_wrap(var, self.quantizer, self.precision)

    def count_up_to(self, *args, **kwargs):
        return self.latent_variable.count_up_to(*args, **kwargs)

    def load(self, *args, **kwargs):
        return self.latent_variable.load(*args, **kwargs)

    @property
    def dtype(self):
        return self.latent_variable.dtype

    @property
    def name(self):
        return self.latent_variable.name

    @property
    def _shared_name(self):
        return self.latent_variable._shared_name

    @property
    def initializer(self):
        return self.latent_variable.initializer

    @property
    def device(self):
        return self.latent_variable.device

    @property
    def op(self):
        return self.latent_variable.op

    @property
    def graph(self):
        return self.latent_variable.graph

    @property
    def shape(self):
        return self.latent_variable.shape

    def get_shape(self):
        return self.latent_variable.get_shape()

    def _gather_saveables_for_checkpoint(self):
        # By delegating this method to the wrapped variable, checkpoints with
        # QuantizedVariables are identical to checkpoints with normal variables.
        # Therefore models checkpointed with QuantizedVariables can be restored on
        # models with normal variables, and vice versa.
        return self.latent_variable._gather_saveables_for_checkpoint()

    # TODO: Maybe encode the fact the variable is an QuantizedVariable in to_proto().
    def to_proto(self, *args, **kwargs):
        return self.latent_variable.to_proto(*args, **kwargs)

    def from_proto(self, *args, **kwargs):
        return self.latent_variable.from_proto(*args, **kwargs)

    def _as_graph_element(self):
        return self._quantize(self.latent_variable._as_graph_element())


QuantizedVariable._OverloadAllOperators()
tf.register_tensor_conversion_function(
    QuantizedVariable, QuantizedVariable._dense_var_to_tensor
)
ops.register_dense_tensor_like_type(QuantizedVariable)


def create_quantized_variable(variable, quantizer=None, precision=None):
    """Creates a QuantizedVariable that wraps another variable.

    This typically just returns `QuantizedVariable(variable)`. But, if the variable
    is a DistributedVariable or one of its subclasses, we instead dynamically
    create a class that subclasses from both QuantizedVariable and
    variable.__class__. This is so the returned variable will still pass
    `isinstance(variable, variable.__class__)`, which is required for
    DistributedVariables and its subclasses to work properly.

    # Arguments
    variable: A floating-point resource variable to wrap.
    quantizer: An optional quantizer to transform the floating-point variable to a
        fake quantized variable.
    precision: An optional integer defining the precision of the quantized variable.
        If `None`, `quantizer.precision` is used.

    # Returns
    A QuantizedVariable that wraps the variable.
    """
    if not isinstance(variable, DistributedVariable):  # type: ignore
        return QuantizedVariable(variable, quantizer, precision)

    class QuantizedDistributedVariable(QuantizedVariable, variable.__class__):
        """A QuantizedVariable that also subclasses from DistributedVariable."""

        def get(self, *args, **kwargs):
            # For some reason this is needed to make unit `x + x` pass on TF 1.14
            return self._quantize(self.latent_variable.get(*args, **kwargs))

    return QuantizedDistributedVariable(variable, quantizer, precision)


def _maybe_wrap(variable, quantizer, precision, wrap=True):
    """Creates an QuantizedVariable that wraps another variable if applicable.

    This function is used to wrap the return value of QuantizedVariable.assign.
    Unfortunately MirroredVariable.assign will (incorrectly) return a Mirrored
    value instead of a MirroredVariable. So we cannot properly wrap it in an
    AutoCastVariable. We return the original variable in that case.

    # Arguments
    variable: A tf.Variable or op.
    quantizer: An optional quantizer to transform the floating-point variable to a
        fake quantized variable.
    precision: An optional integer defining the precision of the quantized variable.
        If `None`, `quantizer.precision` is used.
    wrap: A boolean to define whether to wrap the variable in an QuantizedVariable.

    # Returns
    An QuantizedVariable if wrap is True and variable is a resource variable.
    """
    if wrap and resource_variable_ops.is_resource_variable(variable):
        return create_quantized_variable(variable, quantizer, precision)
    return variable


def _get_name(obj):
    try:
        return obj.__name__
    except AttributeError:
        return obj.__class__.__name__
