from ..dsl.production import ConcreteProduction
from ..dsl.dsl import DSL
from ..types.type import AtomicType
from ..types.type_signature import ConcreteTypeSignature


int_type = AtomicType("int")

basic_arith_dsl = DSL(
    [
        ConcreteProduction(
            "+",
            ConcreteTypeSignature([int_type, int_type], int_type),
            lambda x, y: x + y,
        ),
        ConcreteProduction(
            "1",
            ConcreteTypeSignature([], int_type),
            lambda: 1,
        ),
    ]
)
