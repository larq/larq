"""Contains QuantizedVariable, a variable that can be quantized in the forward pass."""
from tensorflow.python.distribute import values as distribute_values
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from functools import wraps


def quantize(method):
    """A decorator that can quantize the return value of the classmethod.

    Syntactic sugar for `self.quantizer(self.method(*args, **kwargs))`.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        value = method(self, *args, **kwargs)
        if self.quantizer:
            return self.quantizer(value)
        return value

    return wrapper


class QuantizedVariable(variables.Variable):
    """Variable that can be quantized in the forward pass in applicable contexts."""

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
                "variable must be of type tf.ResourceVariable, but got: "
                "%s" % variable
            )
        self.latent_variable = variable
        self.quantizer = quantizer
        self.precision = precision or getattr(quantizer, "precision", None)

    @quantize
    def value(self):
        return self.latent_variable.value()

    @quantize
    def read_value(self):
        return self.latent_variable.read_value()

    @quantize
    def sparse_read(self, indices, name=None):
        """Reads the value of this variable sparsely, using `gather`."""
        return self.latent_variable.sparse_read(indices, name=name)

    @quantize
    def gather_nd(self, indices, name=None):
        """Gather slices of the variable into a Tensor."""
        return self.latent_variable.gather_nd(indices, name=name)

    def __getattr__(self, name):
        return getattr(self.latent_variable, name)

    @quantize
    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        """Converts this variable to a tensor."""
        return self.latent_variable._dense_var_to_tensor(dtype, name, as_ref)

    def eval(self, session=None):
        if self.quantizer:
            return self.quantizer(self.latent_variable).eval(session=session)
        return self.latent_variable.eval(session=session)

    @quantize
    def initialized_value(self):
        return self.latent_variable.initialized_value()

    @property
    @quantize
    def initial_value(self):
        return self.latent_variable.initial_value

    def _should_act_as_resource_variable(self):
        """Pass resource_variable_ops.is_resource_variable check."""
        pass

    def __repr__(self):
        if context.executing_eagerly() and not self._in_graph_mode:
            return (
                f"<{self.__class__.__name__} '{self.name}' shape={self.shape} "
                f"dtype={self.dtype.name} quantizer={self.quantizer.__repr__()} "
                f"precision={self.precision} "
                f"numpy={ops.numpy_text(self.read_value(), is_repr=True)}>"
            )
        return (
            f"<{self.__class__.__name__} '{self.name}' shape={self.shape} "
            f"dtype={self.dtype.name} quantizer={self.quantizer.__repr__()} "
            f"precision={self.precision}>"
        )

    # Method delegations: We delegate the following methods to self.latent_variable.
    # Each of these methods simply calls the same method on self.latent_variable. The
    # base Variable raises NotImplementedError for most of these, so we must
    # override them.
    #
    # We do not define the following methods from Variable for the following
    # reasons:
    #   * 'count_up_to': This method only applies to int variables, which cannot
    #     be wrapped with an QuantizedVariable.
    #   * 'experimental_ref': Instead we inherit the definition from Variable.
    #     If we defined and delegated to Variable, the ref of an QuantizedVariable
    #     would be the same as the ref of the underlying variable, which would be
    #     strange as they are different Python objects.

    def set_shape(self, shape):
        return self.latent_variable.set_shape(self, shape)

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
        return self.latent_variable.assign(value, use_locking, name, read_value)

    def assign_add(self, delta, use_locking=None, name=None, read_value=True):
        return self.latent_variable.assign_add(delta, use_locking, name, read_value)

    def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
        return self.latent_variable.assign_sub(delta, use_locking, name, read_value)

    def scatter_sub(self, sparse_delta, use_locking=False, name=None):
        return self.latent_variable.scatter_sub(sparse_delta, use_locking, name)

    def scatter_add(self, sparse_delta, use_locking=False, name=None):
        return self.latent_variable.scatter_add(sparse_delta, use_locking, name)

    def scatter_max(self, sparse_delta, use_locking=False, name=None):
        return self.latent_variable.scatter_max(sparse_delta, use_locking, name)

    def scatter_min(self, sparse_delta, use_locking=False, name=None):
        return self.latent_variable.scatter_min(sparse_delta, use_locking, name)

    def scatter_mul(self, sparse_delta, use_locking=False, name=None):
        return self.latent_variable.scatter_mul(sparse_delta, use_locking, name)

    def scatter_div(self, sparse_delta, use_locking=False, name=None):
        return self.latent_variable.scatter_div(sparse_delta, use_locking, name)

    def scatter_update(self, sparse_delta, use_locking=False, name=None):
        return self.latent_variable.scatter_update(sparse_delta, use_locking, name)

    def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
        return self.latent_variable.batch_scatter_update(
            sparse_delta, use_locking, name
        )

    def scatter_nd_sub(self, indices, updates, name=None):
        return self.latent_variable.scatter_nd_sub(indices, updates, name)

    def scatter_nd_add(self, indices, updates, name=None):
        return self.latent_variable.scatter_nd_add(indices, updates, name)

    def scatter_nd_update(self, indices, updates, name=None):
        return self.latent_variable.scatter_nd_update(indices, updates, name)

    def count_up_to(self, limit):
        return self.latent_variable.count_up_to(limit)

    def load(self, value, session=None):
        return self.latent_variable.load(value, session)

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
    def to_proto(self, export_scope=None):
        return self.latent_variable.to_proto(export_scope)

    def from_proto(self, variable_def, import_scope=None):
        return self.latent_variable.from_proto(variable_def, import_scope)


QuantizedVariable._OverloadAllOperators()
ops.register_tensor_conversion_function(
    QuantizedVariable, QuantizedVariable._dense_var_to_tensor
)
ops.register_dense_tensor_like_type(QuantizedVariable)


def create_quantized_variable(variable, quantizer=None):
    """Creates an QuantizedVariable that wraps another variable.

    This typically just returns `QuantizedVariable(variable)`. But, if the variable
    is a DistributedVariable or one of its subclasses, we instead dynamically
    create a class that subclasses from both QuantizedVariable and
    variable.__class__. This is so the returned variable will still pass
    `isinstance(variable, variable.__class__)`, which is required for
    DistributedVariables and its subclasses to work properly.

    Args:
      variable: A floating-point resource variable to wrap.

    Returns:
      An QuantizedVariable that wraps the variable.
    """
    if not isinstance(variable, distribute_values.DistributedVariable):
        return QuantizedVariable(variable, quantizer=quantizer)

    class QuantizedDistributedVariable(QuantizedVariable, variable.__class__):
        """An QuantizedVariable that also subclasses from DistributedVariable."""

        def __init__(self, maybe_variable, *args, quantizer=None, **kwargs):
            if not args and not kwargs:
                # The common case: We call the super constructor with a single argument,
                # which is a variable.
                super().__init__(maybe_variable, quantizer=quantizer)
            else:
                # This 'else' branch is needed, as distribution strategies sometimes
                # clone a distributed variable by doing the following:
                #
                #    var = type(var)(var._distribute_strategy, var._device_map, ...)
                #
                # In this case, `maybe_variable` will instead be a distribution
                # strategy. We create the DistributedVariable before wrapping it.
                distribution_strategy = maybe_variable
                inner_var = variable.__class__(distribution_strategy, *args, **kwargs)
                super().__init__(inner_var, quantizer=quantizer)

    return QuantizedDistributedVariable(variable, quantizer=quantizer)
