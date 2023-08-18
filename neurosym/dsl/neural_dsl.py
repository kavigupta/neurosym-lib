from dataclasses import dataclass
from typing import Dict

from neurosym.types.type_signature import ConcreteTypeSignature

from ..programs.hole import Hole
from ..programs.s_expression import InitializedSExpression, SExpression
from ..types.type import AtomicType, Type, ArrowType
from torch import nn

from .production import ConcreteProduction, Production, ParameterizedProduction
from .dsl import DSL


@dataclass
class NeuralDSL(DSL):
    """
    A neural DSL extends `DSL` to handle neural heuristics (ie: type-appropriate NN productions)
    These neural heuristics can be used to fill holes in partial programs.
    Required to run NEAR.

    :TODO I'm electing to keep `partial_productions` separate from `productions` for now. We might
    want to seek a more elegant solution in the future.
    """
    partial_programs: Dict[Type, SExpression]
    @classmethod
    def from_dsl(cls, dsl: DSL, partial_modules: Dict[Type, nn.Module]):
        """
        We cannot directly add neural models since our language is pure and functional.
        Our workaround is to construct two production rules:
         -  a "construct prod" for intializing each model which contains the state info.
         - a "application prod" that can only be filled with "construct prod" for that model.
        ie:
        module_c : [] -> module_c_type
        module_app : (module_inp, module_c_type) -> module_out

        These rules will be added to the DSL.
        Concurently, we will then define Sexpression to call each of these rules.
        ie:
            SExpr(module_app module_inp (module_c) -> module_out)
        Our holes will be replaced with the appropriate SExpr.
        """
        partial_productions = []
        partial_programs = {}
        
        for i, (fn_type, module) in enumerate(partial_modules.items()):
            assert isinstance(fn_type, ArrowType), f"Type of partial NN module must be an ArrowType, got {fn_type}"
            # @TODO[AS]: This is VERY hacky. Need a formal way to segregate partial modules.
            identifier = "partial{name}_{idx}".format(idx=i, name=module.config.model_name)
            module_obj_type = AtomicType(identifier)
            module_c_prod = ParameterizedProduction(
                identifier + "_c",
                ConcreteTypeSignature([], module_obj_type),
                lambda f_module: f_module,
                dict(f_module= lambda: module),
            )

            def module_app(module, **inputs):
                return module(**inputs)

            module_app_prod = ConcreteProduction(
                identifier + "_app",
                ConcreteTypeSignature([module_obj_type, *fn_type.input_type], fn_type.output_type),
                module_app,
            )
            partial_productions.append(module_c_prod)
            partial_productions.append(module_app_prod)

            # @TODO[AS]: Need to define what the SExpr will be.
            raise NotImplementedError("TODO: Need to define what the SExpr will be.")
            partial_programs[fn_type].append(lambda : InitializedSExpression(
                    # program.symbol,
                    # tuple(self.initialize(child) for child in program.children),
                    # prod.initialize(),
                ))

        productions = dsl.productions + partial_productions

        return cls(productions=productions, partial_programs=partial_programs)

    def get_partial_program(self, hole: Hole) -> Production:
        matching_types = list(filter(lambda type: hole.type == type, self.partial_programs.keys()))

        if len(matching_types) == 0:
            raise ValueError(f"No partial production found for type {hole.type}")
        elif len(matching_types) > 1:
            raise ValueError(f"Multiple partial productions found for type {hole.type}")

        return self.partial_programs[matching_types[0]]

    def initialize(self, program: SExpression) -> InitializedSExpression:
        """
        Initializes all the productions in the given program.

        Returns a new program with the same structure, but with all the productions
        initialized.
        """
        if isinstance(program, Hole):
            prod =  self.get_partial_program(program)
        else:
            prod = self.get_production(program.symbol)
        return InitializedSExpression(
            program.symbol,
            tuple(self.initialize(child) for child in program.children),
            prod.initialize(),
        )