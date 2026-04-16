from neurosym.dsl.production import LambdaProduction
from neurosym.types.type import ArrowType

from .type_preorder_mask import TypePreorderMask, _UnionTypeWithEnvironment


class TypePreorderMaskELF(TypePreorderMask):
    """
    Like ``ns.TypePreorderMask``, but only allows lambda productions to be used
    when the expected type is a function type. This follows the Eta-Long Form
    (ELF) convention for representing programs, specifically the one used
    by DreamCoder; where programs such as ``(car (cons (lam $0) nil))`` are
    disallowed as ``car`` is returning a function type, which is not allowed.
    """

    def valid_productions(self, twe):
        productions = super().valid_productions(twe)
        if isinstance(twe, _UnionTypeWithEnvironment):
            # Apply ELF constraint only when every type in the union is a
            # function type; if even one is not, non-lambda productions may
            # legitimately fill that position.
            if all(isinstance(t.typ, ArrowType) for t in twe.types):
                productions = [
                    prod for prod in productions if isinstance(prod, LambdaProduction)
                ]
        elif isinstance(twe.typ, ArrowType):
            productions = [
                prod for prod in productions if isinstance(prod, LambdaProduction)
            ]
        return productions
