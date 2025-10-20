from neurosym.dsl.production import LambdaProduction
from neurosym.types.type import ArrowType

from .type_preorder_mask import TypePreorderMask


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
        if isinstance(twe.typ, ArrowType):
            productions = [
                prod for prod in productions if isinstance(prod, LambdaProduction)
            ]
        return productions
