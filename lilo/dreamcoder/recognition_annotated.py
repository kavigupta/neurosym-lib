from dreamcoder.enumeration import *
from dreamcoder.grammar import *
from dreamcoder.utilities import RunWithTimeout

# luke


import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

import numpy as np

# luke
import json


def variable(x : list | np.ndarray | torch.Tensor, volatile=False, cuda=False):
    """
    Converts x into a PyTorch Variable. If x is a list, converts it into a numpy array first.

    Note: Variables are being deprecated in newer PyTorch versions.

    Args:
        x (array-like): An array-like object for conversion into a Variable.
        volatile (bool, optional): Set to true if inference-only and no backpropagation. Defaults to False.
        cuda (bool, optional): Converts x into a cuda tensor if x is a tensor. Defaults to False.

    Returns:
        Variable: A PyTorch Variable
    """

    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, (np.ndarray, np.generic)):
        x = torch.from_numpy(x)
    if cuda:
        x = x.cuda()
    return Variable(x, volatile=volatile)


def maybe_cuda(x: torch.Tensor, use_cuda: bool):
    """
    Uses cuda version of x if use_cuda is True, otherwise uses the cpu version.

    Args:
        x (torch.Tensor): A tensor to be converted to cuda.
        use_cuda (bool): A boolean flag to determine whether to use cuda or not.

    Returns:
        torch.Tensor: A tensor that is either on the cpu or cuda.
    """
    if use_cuda:
        return x.cuda()
    else:
        return x


def is_torch_not_a_number(v: torch.Tensor):
    """
    Checks whether a tortured variable is nan.
    
    Args:
        v (torch.Tensor): A tensor to be checked for nan.
    Returns:
        bool: A boolean flag indicating whether the tensor is nan.
    """
    v = v.data
    if not ((v == v).item()):
        return True
    return False


def is_torch_invalid(v : torch.Tensor):
    """
    Checks whether a torch variable is nan or inf

    Args:
        v (torch.Tensor): A tensor to be checked for nan or inf.
    Returns:
        bool: A boolean flag indicating whether the tensor is nan or inf.

    """
    if is_torch_not_a_number(v):
        return True
    a = v - v
    if is_torch_not_a_number(a):
        return True
    return False


def _relu(x : torch.Tensor):
    """
    Applies the rectified linear unit function to x.

    Args:
        x (torch.Tensor): A tensor to which the rectified linear unit function is applied.

    Returns:
        torch.Tensor: A tensor with the rectified linear unit function applied.
    """
    return x.clamp(min=0)


class Entropy(nn.Module):
    def __init__(self):
        """
        Initializes the Entropy class.
        """
        super(Entropy, self).__init__()

    def forward(self, x: torch.Tensor):
        """
        Calculates the entropy of x.

        Args:
            x (torch.Tensor): A tensor for which the entropy is calculated.

        Returns:
            torch.Tensor: A tensor containing the entropy of x.
        """
        b = F.softmax(x, dim=0) * F.log_softmax(x, dim=0)
        b = -1.0 * b.sum()
        return b


class GrammarNetwork(nn.Module):
    """Neural network that outputs a grammar"""

    def __init__(self, inputDimensionality: int, grammar: Grammar):
        """
        Initializes the GrammarNetwork class.

        Args:
            inputDimensionality (int): The dimensionality of the input for the logProductions neural network.
            grammar (Grammar): The grammar on which the neural network is trained.
        """
        super(GrammarNetwork, self).__init__()
        self.logProductions = nn.Linear(inputDimensionality, len(grammar) + 1)
        self.grammar = grammar

    def forward(self, x):
        """
        Takes as input an inputDimensionality-dimensional vector and returns Grammar Tensor-valued probabilities
        
        Args:
            x (torch.Tensor): A inputDimensionality-dimensional tensor input for the neural network.

        Returns:
            Grammar: A Grammar object containing the probabilities of the productions.            

        """
        logProductions = self.logProductions(x)
        return Grammar(
            logProductions[-1].view(1),  # logVariable
            [
                (logProductions[k].view(1), t, program)
                for k, (_, t, program) in enumerate(self.grammar.productions)
            ],
            continuationType=self.grammar.continuationType,
        )

    def batchedLogLikelihoods(self, xs: torch.Tensor, summaries: list):
        """
        Takes as input BxinputDimensionality vector & B likelihood summaries;
        returns B-dimensional vector containing log likelihood of each summary
        
        Args:
            xs: a B x inputDimensionality vector
            summaries: a list of B likelihood summaries
        Returns:
            torch.Tensor: A B-dimensional tensor containing the log likelihood of each summary.

        """
        use_cuda = xs.device.type == "cuda"

        B = xs.size(0)
        assert len(summaries) == B
        logProductions = self.logProductions(xs)

        # uses[b][p] is # uses of primitive p by summary b
        uses = np.zeros((B, len(self.grammar) + 1))
        for b, summary in enumerate(summaries):
            for p, production in enumerate(self.grammar.primitives):
                uses[b, p] = summary.uses.get(production, 0.0)
            uses[b, len(self.grammar)] = summary.uses.get(Index(0), 0)

        numerator = (
            logProductions * maybe_cuda(torch.from_numpy(uses).float(), use_cuda)
        ).sum(1)
        numerator += maybe_cuda(
            torch.tensor([summary.constant for summary in summaries]).float(), use_cuda
        )

        alternativeSet = {normalizer for s in summaries for normalizer in s.normalizers}
        alternativeSet = list(alternativeSet)

        mask = np.zeros((len(alternativeSet), len(self.grammar) + 1))
        for tau in range(len(alternativeSet)):
            for p, production in enumerate(self.grammar.primitives):
                mask[tau, p] = (
                    0.0 if production in alternativeSet[tau] else NEGATIVEINFINITY
                )
            mask[tau, len(self.grammar)] = (
                0.0 if Index(0) in alternativeSet[tau] else NEGATIVEINFINITY
            )
        mask = maybe_cuda(torch.tensor(mask).float(), use_cuda)

        # mask: Rx|G|
        # logProductions: Bx|G|
        # Want: mask + logProductions : BxRx|G| = z
        z = mask.repeat(B, 1, 1) + logProductions.repeat(
            len(alternativeSet), 1, 1
        ).transpose(1, 0)
        # z: BxR
        z = torch.logsumexp(z, 2)  # pytorch 1.0 dependency

        # Calculate how many times each normalizer was used
        N = np.zeros((B, len(alternativeSet)))
        for b, summary in enumerate(summaries):
            for tau, alternatives in enumerate(alternativeSet):
                N[b, tau] = summary.normalizers.get(alternatives, 0.0)

        denominator = (maybe_cuda(torch.tensor(N).float(), use_cuda) * z).sum(1)
        return numerator - denominator


class ContextualGrammarNetwork_LowRank(nn.Module):
    def __init__(self, inputDimensionality: int, grammar: Grammar, R: int = 16):
        """
        
        Low-rank approximation to bigram model. Parameters is linear in number of primitives.
        
        Args:
            inputDimensionality: dimensionality of input for logProductions neural network.
            grammar: grammar on which neural network is trained.
            R: maximum rank (embedding size)
        
        """

        super(ContextualGrammarNetwork_LowRank, self).__init__()

        self.grammar = grammar

        self.R = R  # embedding size

        # library now just contains a list of indicies which go with each primitive
        self.grammar = grammar
        self.library = {}
        self.n_grammars = 0
        for prim in grammar.primitives:
            numberOfArguments = len(prim.infer().functionArguments())
            idx_list = list(range(self.n_grammars, self.n_grammars + numberOfArguments))
            self.library[prim] = idx_list
            self.n_grammars += numberOfArguments

        # We had an extra grammar for when there is no parent and for when the parent is a variable
        self.n_grammars += 2
        self.transitionMatrix = LowRank(
            inputDimensionality, self.n_grammars, len(grammar) + 1, R
        )

    def grammarFromVector(self, logProductions: torch.Tensor):

        """
        Produces a Grammar object from a vector of log probabilities of productions.

        Args:
            logProductions(torch.Tensor): A len(grammar)+1 tensor of log probabilities of productions.
        Returns:
            Grammar: A Grammar object containing the log probabilities of the productions.
        """

        return Grammar(
            logProductions[-1].view(1),
            [
                (logProductions[k].view(1), t, program)
                for k, (_, t, program) in enumerate(self.grammar.productions)
            ],
            continuationType=self.grammar.continuationType,
        )

    def forward(self, x: torch.Tensor):

        """
        A forward pass of the bigram model neural network.

        Args:
            x: A inputDimensionality-dimensional tensor input for the neural network.
        Returns:
            ContextualGrammar: A ContextualGrammar object containing the probabilities of each primitive in the library.
        """

        assert (
            len(x.size()) == 1
        ), "contextual grammar doesn't currently support batching"

        transitionMatrix = self.transitionMatrix(x)

        return ContextualGrammar(
            self.grammarFromVector(transitionMatrix[-1]),
            self.grammarFromVector(transitionMatrix[-2]),
            {
                prim: [self.grammarFromVector(transitionMatrix[j]) for j in js]
                for prim, js in self.library.items()
            },
        )

    def vectorizedLogLikelihoods(self, x: torch.Tensor, summaries: list):
        """

        Calculates the log likelihood of a batch of summaries given a batch of input tensors.

        Args:
            x (torch.Tensor): An inputDimensionality tensor input for the neural network.
            summaries (list): A list of B likelihood summaries.

        Note: This asserts False so it may not be used by DreamCoder's current codebase.
        """
        B = len(summaries)
        G = len(self.grammar) + 1

        # Which column of the transition matrix corresponds to which primitive
        primitiveColumn = {
            p: c for c, (_1, _2, p) in enumerate(self.grammar.productions)
        }
        primitiveColumn[Index(0)] = G - 1
        # Which row of the transition matrix corresponds to which context
        contextRow = {
            (parent, index): r
            for parent, indices in self.library.items()
            for index, r in enumerate(indices)
        }
        contextRow[(None, None)] = self.n_grammars - 1
        contextRow[(Index(0), None)] = self.n_grammars - 2

        transitionMatrix = self.transitionMatrix(x)

        # uses[b][g][p] is # uses of primitive p by summary b for parent g
        uses = np.zeros((B, self.n_grammars, len(self.grammar) + 1))
        for b, summary in enumerate(summaries):
            for e, ss in summary.library.items():
                for g, s in zip(self.library[e], ss):
                    assert g < self.n_grammars - 2
                    for p, production in enumerate(self.grammar.primitives):
                        uses[b, g, p] = s.uses.get(production, 0.0)
                    uses[b, g, len(self.grammar)] = s.uses.get(Index(0), 0)

            # noParent: this is the last network output
            for p, production in enumerate(self.grammar.primitives):
                uses[b, self.n_grammars - 1, p] = summary.noParent.uses.get(
                    production, 0.0
                )
            uses[b, self.n_grammars - 1, G - 1] = summary.noParent.uses.get(
                Index(0), 0.0
            )

            # variableParent: this is the penultimate network output
            for p, production in enumerate(self.grammar.primitives):
                uses[b, self.n_grammars - 2, p] = summary.variableParent.uses.get(
                    production, 0.0
                )
            uses[b, self.n_grammars - 2, G - 1] = summary.variableParent.uses.get(
                Index(0), 0.0
            )

        uses = maybe_cuda(torch.tensor(uses).float(), use_cuda)
        numerator = uses.view(B, -1) @ transitionMatrix.view(-1)

        constant = np.zeros(B)
        for b, summary in enumerate(summaries):
            constant[b] += summary.noParent.constant + summary.variableParent.constant
            for ss in summary.library.values():
                for s in ss:
                    constant[b] += s.constant

        numerator = numerator + maybe_cuda(torch.tensor(constant).float(), use_cuda)

        # Calculate the god-awful denominator
        # Map from (parent, index, {set-of-alternatives}) to [occurrences-in-summary-zero, occurrences-in-summary-one, ...]
        alternativeSet = {}
        for b, summary in enumerate(summaries):
            for normalizer, frequency in summary.noParent.normalizers.items():
                k = (None, None, normalizer)
                alternativeSet[k] = alternativeSet.get(k, np.zeros(B))
                alternativeSet[k][b] += frequency
            for normalizer, frequency in summary.variableParent.normalizers.items():
                k = (Index(0), None, normalizer)
                alternativeSet[k] = alternativeSet.get(k, np.zeros(B))
                alternativeSet[k][b] += frequency
            for parent, ss in summary.library.items():
                for argumentIndex, s in enumerate(ss):
                    for normalizer, frequency in s.normalizers.items():
                        k = (parent, argumentIndex, normalizer)
                        alternativeSet[k] = alternativeSet.get(k, zeros(B))
                        alternativeSet[k][b] += frequency

        # Calculate each distinct normalizing constant
        alternativeNormalizer = {}
        for parent, index, alternatives in alternativeSet:
            r = transitionMatrix[contextRow[(parent, index)]]
            entries = r[[primitiveColumn[alternative] for alternative in alternatives]]
            alternativeNormalizer[(parent, index, alternatives)] = torch.logsumexp(
                entries, dim=0
            )

        # Concatenate the normalizers into a vector
        normalizerKeys = list(alternativeSet.keys())
        normalizerVector = torch.cat([alternativeNormalizer[k] for k in normalizerKeys])

        assert False, "This function is still in progress."

    def batchedLogLikelihoods(self, xs: torch.Tensor, summaries: list):
        """
        Takes as input B x inputDimensionality vector & B likelihood summaries;
        returns B-dimensional vector containing log likelihood of each summary
        
        Args:
            xs (torch.Tensor): a B x inputDimensionality vector
            summaries (list): a list of B likelihood summaries

        Returns:
            torch.Tensor: A B-dimensional tensor containing the log likelihood of each summary.
        """
        use_cuda = xs.device.type == "cuda"

        B = xs.shape[0]
        G = len(self.grammar) + 1
        assert len(summaries) == B

        # logProductions: Bx n_grammars x G
        logProductions = self.transitionMatrix(xs)
        # uses[b][g][p] is # uses of primitive p by summary b for parent g
        uses = np.zeros((B, self.n_grammars, len(self.grammar) + 1))
        for b, summary in enumerate(summaries):
            for e, ss in summary.library.items():
                for g, s in zip(self.library[e], ss):
                    assert g < self.n_grammars - 2
                    for p, production in enumerate(self.grammar.primitives):
                        uses[b, g, p] = s.uses.get(production, 0.0)
                    uses[b, g, len(self.grammar)] = s.uses.get(Index(0), 0)

            # noParent: this is the last network output
            for p, production in enumerate(self.grammar.primitives):
                uses[b, self.n_grammars - 1, p] = summary.noParent.uses.get(
                    production, 0.0
                )
            uses[b, self.n_grammars - 1, G - 1] = summary.noParent.uses.get(
                Index(0), 0.0
            )

            # variableParent: this is the penultimate network output
            for p, production in enumerate(self.grammar.primitives):
                uses[b, self.n_grammars - 2, p] = summary.variableParent.uses.get(
                    production, 0.0
                )
            uses[b, self.n_grammars - 2, G - 1] = summary.variableParent.uses.get(
                Index(0), 0.0
            )

        numerator = (
            (logProductions * maybe_cuda(torch.tensor(uses).float(), use_cuda))
            .view(B, -1)
            .sum(1)
        )

        constant = np.zeros(B)
        for b, summary in enumerate(summaries):
            constant[b] += summary.noParent.constant + summary.variableParent.constant
            for ss in summary.library.values():
                for s in ss:
                    constant[b] += s.constant

        numerator += maybe_cuda(torch.tensor(constant).float(), use_cuda)

        if True:

            # Calculate the god-awful denominator
            alternativeSet = set()
            for summary in summaries:
                for normalizer in summary.noParent.normalizers:
                    alternativeSet.add(normalizer)
                for normalizer in summary.variableParent.normalizers:
                    alternativeSet.add(normalizer)
                for ss in summary.library.values():
                    for s in ss:
                        for normalizer in s.normalizers:
                            alternativeSet.add(normalizer)
            alternativeSet = list(alternativeSet)

            mask = np.zeros((len(alternativeSet), G))
            for tau in range(len(alternativeSet)):
                for p, production in enumerate(self.grammar.primitives):
                    mask[tau, p] = (
                        0.0 if production in alternativeSet[tau] else NEGATIVEINFINITY
                    )
                mask[tau, G - 1] = (
                    0.0 if Index(0) in alternativeSet[tau] else NEGATIVEINFINITY
                )
            mask = maybe_cuda(torch.tensor(mask).float(), use_cuda)

            z = mask.repeat(self.n_grammars, 1, 1).repeat(
                B, 1, 1, 1
            ) + logProductions.repeat(len(alternativeSet), 1, 1, 1).transpose(
                0, 1
            ).transpose(
                1, 2
            )
            z = torch.logsumexp(z, 3)  # pytorch 1.0 dependency

            N = np.zeros((B, self.n_grammars, len(alternativeSet)))
            for b, summary in enumerate(summaries):
                for e, ss in summary.library.items():
                    for g, s in zip(self.library[e], ss):
                        assert g < self.n_grammars - 2
                        for r, alternatives in enumerate(alternativeSet):
                            N[b, g, r] = s.normalizers.get(alternatives, 0.0)
                # noParent: this is the last network output
                for r, alternatives in enumerate(alternativeSet):
                    N[b, self.n_grammars - 1, r] = summary.noParent.normalizers.get(
                        alternatives, 0.0
                    )
                # variableParent: this is the penultimate network output
                for r, alternatives in enumerate(alternativeSet):
                    N[
                        b, self.n_grammars - 2, r
                    ] = summary.variableParent.normalizers.get(alternatives, 0.0)
            N = maybe_cuda(torch.tensor(N).float(), use_cuda)
            denominator = (N * z).sum(1).sum(1)
        else:
            gs = [self(xs[b]) for b in range(B)]
            denominator = torch.cat(
                [summary.denominator(g) for summary, g in zip(summaries, gs)]
            )

        ll = numerator - denominator

        if False:  # verifying that batching works correctly
            gs = [self(xs[b]) for b in range(B)]
            _l = torch.cat(
                [summary.logLikelihood(g) for summary, g in zip(summaries, gs)]
            )
            assert torch.all((ll - _l).abs() < 0.0001)
        return ll


class ContextualGrammarNetwork_Mask(nn.Module):
    def __init__(self, inputDimensionality: int, grammar: Grammar):
        """
        Bigram model, but where the bigram transitions are unconditional.
        Individual primitive probabilities are still conditional (predicted by neural network)

        Args:
            inputDimensionality (int): dimensionality of input for logProductions neural network.
            grammar (Grammar): grammar on which neural network is trained.

        """

        super(ContextualGrammarNetwork_Mask, self).__init__()

        self.grammar = grammar

        # library now just contains a list of indicies which go with each primitive
        self.grammar = grammar
        self.library = {}
        self.n_grammars = 0
        for prim in grammar.primitives:
            numberOfArguments = len(prim.infer().functionArguments())
            idx_list = list(range(self.n_grammars, self.n_grammars + numberOfArguments))
            self.library[prim] = idx_list
            self.n_grammars += numberOfArguments

        # We had an extra grammar for when there is no parent and for when the parent is a variable
        self.n_grammars += 2
        self._transitionMatrix = nn.Parameter(
            nn.init.xavier_uniform(torch.Tensor(self.n_grammars, len(grammar) + 1))
        )
        self._logProductions = nn.Linear(inputDimensionality, len(grammar) + 1)

    def transitionMatrix(self, x: torch.Tensor):
        """
        Calculates the transition matrix for the neural network.

        Args:
            x (torch.Tensor): A inputDimensionality-dimensional tensor input for the neural network.

        Returns:
            torch.Tensor: A tensor containing the transition matrix.
        """
        if len(x.shape) == 1:  # not batched
            return self._logProductions(x) + self._transitionMatrix  # will broadcast
        elif len(x.shape) == 2:  # batched
            return self._logProductions(x).unsqueeze(1).repeat(
                1, self.n_grammars, 1
            ) + self._transitionMatrix.unsqueeze(0).repeat(x.size(0), 1, 1)
        else:
            assert False, "unknown shape for transition matrix input"

    def grammarFromVector(self, logProductions: torch.Tensor):
        """
         Produces a Grammar object from a vector of log probabilities of productions.

        Args:
            logProductions(torch.Tensor): A len(grammar)+1 tensor of log probabilities of productions.
        Returns:
            Grammar: A Grammar object containing the log probabilities of the productions.
        """
        return Grammar(
            logProductions[-1].view(1),
            [
                (logProductions[k].view(1), t, program)
                for k, (_, t, program) in enumerate(self.grammar.productions)
            ],
            continuationType=self.grammar.continuationType,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the bigram (with unconditional probabilities) neural network.

        Args:
            x (torch.Tensor): A inputDimensionality-dimensional tensor input for the neural network.

        Returns:
            ContextualGrammar: A ContextualGrammar object containing the probabilities of each primitive in the library.
        """
        assert (
            len(x.size()) == 1
        ), "contextual grammar doesn't currently support batching"

        transitionMatrix = self.transitionMatrix(x)

        return ContextualGrammar(
            self.grammarFromVector(transitionMatrix[-1]),
            self.grammarFromVector(transitionMatrix[-2]),
            {
                prim: [self.grammarFromVector(transitionMatrix[j]) for j in js]
                for prim, js in self.library.items()
            },
        )

    def batchedLogLikelihoods(self, xs: torch.Tensor, summaries: list):
        """
        Takes as input BxinputDimensionality vector & B likelihood summaries;
        returns B-dimensional vector containing log likelihood of each summary
        
        Args:
            xs (torch.Tensor): a B x inputDimensionality vector
            summaries (list): a list of B likelihood summaries
        Returns:
            torch.Tensor: A B-dimensional tensor containing the log likelihood of each summary.
        """
        use_cuda = xs.device.type == "cuda"

        B = xs.shape[0]
        G = len(self.grammar) + 1
        assert len(summaries) == B

        # logProductions: Bx n_grammars x G
        logProductions = self.transitionMatrix(xs)
        # uses[b][g][p] is # uses of primitive p by summary b for parent g
        uses = np.zeros((B, self.n_grammars, len(self.grammar) + 1))
        for b, summary in enumerate(summaries):
            for e, ss in summary.library.items():
                for g, s in zip(self.library[e], ss):
                    assert g < self.n_grammars - 2
                    for p, production in enumerate(self.grammar.primitives):
                        uses[b, g, p] = s.uses.get(production, 0.0)
                    uses[b, g, len(self.grammar)] = s.uses.get(Index(0), 0)

            # noParent: this is the last network output
            for p, production in enumerate(self.grammar.primitives):
                uses[b, self.n_grammars - 1, p] = summary.noParent.uses.get(
                    production, 0.0
                )
            uses[b, self.n_grammars - 1, G - 1] = summary.noParent.uses.get(
                Index(0), 0.0
            )

            # variableParent: this is the penultimate network output
            for p, production in enumerate(self.grammar.primitives):
                uses[b, self.n_grammars - 2, p] = summary.variableParent.uses.get(
                    production, 0.0
                )
            uses[b, self.n_grammars - 2, G - 1] = summary.variableParent.uses.get(
                Index(0), 0.0
            )

        numerator = (
            (logProductions * maybe_cuda(torch.tensor(uses).float(), use_cuda))
            .view(B, -1)
            .sum(1)
        )

        constant = np.zeros(B)
        for b, summary in enumerate(summaries):
            constant[b] += summary.noParent.constant + summary.variableParent.constant
            for ss in summary.library.values():
                for s in ss:
                    constant[b] += s.constant

        numerator += maybe_cuda(torch.tensor(constant).float(), use_cuda)

        if True:

            # Calculate the god-awful denominator
            alternativeSet = set()
            for summary in summaries:
                for normalizer in summary.noParent.normalizers:
                    alternativeSet.add(normalizer)
                for normalizer in summary.variableParent.normalizers:
                    alternativeSet.add(normalizer)
                for ss in summary.library.values():
                    for s in ss:
                        for normalizer in s.normalizers:
                            alternativeSet.add(normalizer)
            alternativeSet = list(alternativeSet)

            mask = np.zeros((len(alternativeSet), G))
            for tau in range(len(alternativeSet)):
                for p, production in enumerate(self.grammar.primitives):
                    mask[tau, p] = (
                        0.0 if production in alternativeSet[tau] else NEGATIVEINFINITY
                    )
                mask[tau, G - 1] = (
                    0.0 if Index(0) in alternativeSet[tau] else NEGATIVEINFINITY
                )
            mask = maybe_cuda(torch.tensor(mask).float(), use_cuda)

            z = mask.repeat(self.n_grammars, 1, 1).repeat(
                B, 1, 1, 1
            ) + logProductions.repeat(len(alternativeSet), 1, 1, 1).transpose(
                0, 1
            ).transpose(
                1, 2
            )
            z = torch.logsumexp(z, 3)  # pytorch 1.0 dependency

            N = np.zeros((B, self.n_grammars, len(alternativeSet)))
            for b, summary in enumerate(summaries):
                for e, ss in summary.library.items():
                    for g, s in zip(self.library[e], ss):
                        assert g < self.n_grammars - 2
                        for r, alternatives in enumerate(alternativeSet):
                            N[b, g, r] = s.normalizers.get(alternatives, 0.0)
                # noParent: this is the last network output
                for r, alternatives in enumerate(alternativeSet):
                    N[b, self.n_grammars - 1, r] = summary.noParent.normalizers.get(
                        alternatives, 0.0
                    )
                # variableParent: this is the penultimate network output
                for r, alternatives in enumerate(alternativeSet):
                    N[
                        b, self.n_grammars - 2, r
                    ] = summary.variableParent.normalizers.get(alternatives, 0.0)
            N = maybe_cuda(torch.tensor(N).float(), use_cuda)
            denominator = (N * z).sum(1).sum(1)
        else:
            gs = [self(xs[b]) for b in range(B)]
            denominator = torch.cat(
                [summary.denominator(g) for summary, g in zip(summaries, gs)]
            )

        ll = numerator - denominator

        if False:  # verifying that batching works correctly
            gs = [self(xs[b]) for b in range(B)]
            _l = torch.cat(
                [summary.logLikelihood(g) for summary, g in zip(summaries, gs)]
            )
            assert torch.all((ll - _l).abs() < 0.0001)
        return ll


class ContextualGrammarNetwork(nn.Module):
    """Like GrammarNetwork but ~contextual~"""

    def __init__(self, inputDimensionality: int, grammar: Grammar):
        """
        Create a Contextual Grammar Network.

        Args:
            inputDimensionality (int): dimensionality of input for logProductions neural network.
            grammar (Grammar): grammar on which neural network is trained.

        """
        super(ContextualGrammarNetwork, self).__init__()

        # library now just contains a list of indicies which go with each primitive
        self.grammar = grammar
        self.library = {}
        self.n_grammars = 0
        for prim in grammar.primitives:
            numberOfArguments = len(prim.infer().functionArguments())
            idx_list = list(range(self.n_grammars, self.n_grammars + numberOfArguments))
            self.library[prim] = idx_list
            self.n_grammars += numberOfArguments

        # We had an extra grammar for when there is no parent and for when the parent is a variable
        self.n_grammars += 2
        self.network = nn.Linear(
            inputDimensionality, (self.n_grammars) * (len(grammar) + 1)
        )

    def grammarFromVector(self, logProductions: torch.Tensor):
        """
        Produces a Grammar object from a vector of log probabilities of productions.

        Args:
            logProductions (torch.Tensor): A len(grammar)+1 tensor of log probabilities of productions.

        Returns:
            Grammar: A Grammar object containing the log probabilities of the productions.
        """
        return Grammar(
            logProductions[-1].view(1),
            [
                (logProductions[k].view(1), t, program)
                for k, (_, t, program) in enumerate(self.grammar.productions)
            ],
            continuationType=self.grammar.continuationType,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the contextual grammar neural network.        

        Args:
            x (torch.Tensor): A inputDimensionality-dimensional tensor input for the neural network.

        Returns:
            ContextualGrammar: A ContextualGrammar object containing the probabilities of each primitive in the library.
        """
        assert (
            len(x.size()) == 1
        ), "contextual grammar doesn't currently support batching"

        allVars = self.network(x).view(self.n_grammars, -1)
        return ContextualGrammar(
            self.grammarFromVector(allVars[-1]),
            self.grammarFromVector(allVars[-2]),
            {
                prim: [self.grammarFromVector(allVars[j]) for j in js]
                for prim, js in self.library.items()
            },
        )

    def batchedLogLikelihoods(self, xs: torch.Tensor, summaries:list):
        """
        Takes as input B x inputDimensionality vector & B likelihood summaries;
        returns B-dimensional vector containing log likelihood of each summary
        
        Args:
            xs: a B x inputDimensionality vector
            summaries: a list of B likelihood summaries
        Returns:
            torch.Tensor: A B-dimensional tensor containing the log likelihood of each summary.
        """

        use_cuda = xs.device.type == "cuda"
        

        B = xs.shape[0]
        G = len(self.grammar) + 1
        assert len(summaries) == B

        # logProductions: Bx n_grammars x G
        logProductions = self.network(xs).view(B, self.n_grammars, G)
        # uses[b][g][p] is # uses of primitive p by summary b for parent g
        uses = np.zeros((B, self.n_grammars, len(self.grammar) + 1))
        for b, summary in enumerate(summaries):
            for e, ss in summary.library.items():
                for g, s in zip(self.library[e], ss):
                    assert g < self.n_grammars - 2
                    for p, production in enumerate(self.grammar.primitives):
                        uses[b, g, p] = s.uses.get(production, 0.0)
                    uses[b, g, len(self.grammar)] = s.uses.get(Index(0), 0)

            # noParent: this is the last network output
            for p, production in enumerate(self.grammar.primitives):
                uses[b, self.n_grammars - 1, p] = summary.noParent.uses.get(
                    production, 0.0
                )
            uses[b, self.n_grammars - 1, G - 1] = summary.noParent.uses.get(
                Index(0), 0.0
            )

            # variableParent: this is the penultimate network output
            for p, production in enumerate(self.grammar.primitives):
                uses[b, self.n_grammars - 2, p] = summary.variableParent.uses.get(
                    production, 0.0
                )
            uses[b, self.n_grammars - 2, G - 1] = summary.variableParent.uses.get(
                Index(0), 0.0
            )

        numerator = (
            (logProductions * maybe_cuda(torch.tensor(uses).float(), use_cuda))
            .view(B, -1)
            .sum(1)
        )

        constant = np.zeros(B)
        for b, summary in enumerate(summaries):
            constant[b] += summary.noParent.constant + summary.variableParent.constant
            for ss in summary.library.values():
                for s in ss:
                    constant[b] += s.constant

        numerator += maybe_cuda(torch.tensor(constant).float(), use_cuda)

        # Calculate the god-awful denominator
        alternativeSet = set()
        for summary in summaries:
            for normalizer in summary.noParent.normalizers:
                alternativeSet.add(normalizer)
            for normalizer in summary.variableParent.normalizers:
                alternativeSet.add(normalizer)
            for ss in summary.library.values():
                for s in ss:
                    for normalizer in s.normalizers:
                        alternativeSet.add(normalizer)
        alternativeSet = list(alternativeSet)

        mask = np.zeros((len(alternativeSet), G))
        for tau in range(len(alternativeSet)):
            for p, production in enumerate(self.grammar.primitives):
                mask[tau, p] = (
                    0.0 if production in alternativeSet[tau] else NEGATIVEINFINITY
                )
            mask[tau, G - 1] = (
                0.0 if Index(0) in alternativeSet[tau] else NEGATIVEINFINITY
            )
        mask = maybe_cuda(torch.tensor(mask).float(), use_cuda)

        z = mask.repeat(self.n_grammars, 1, 1).repeat(
            B, 1, 1, 1
        ) + logProductions.repeat(len(alternativeSet), 1, 1, 1).transpose(
            0, 1
        ).transpose(
            1, 2
        )
        z = torch.logsumexp(z, 3)  # pytorch 1.0 dependency

        N = np.zeros((B, self.n_grammars, len(alternativeSet)))
        for b, summary in enumerate(summaries):
            for e, ss in summary.library.items():
                for g, s in zip(self.library[e], ss):
                    assert g < self.n_grammars - 2
                    for r, alternatives in enumerate(alternativeSet):
                        N[b, g, r] = s.normalizers.get(alternatives, 0.0)
            # noParent: this is the last network output
            for r, alternatives in enumerate(alternativeSet):
                N[b, self.n_grammars - 1, r] = summary.noParent.normalizers.get(
                    alternatives, 0.0
                )
            # variableParent: this is the penultimate network output
            for r, alternatives in enumerate(alternativeSet):
                N[b, self.n_grammars - 2, r] = summary.variableParent.normalizers.get(
                    alternatives, 0.0
                )
        N = maybe_cuda(torch.tensor(N).float(), use_cuda)

        denominator = (N * z).sum(1).sum(1)
        ll = numerator - denominator

        if False:  # verifying that batching works correctly
            gs = [self(xs[b]) for b in range(B)]
            _l = torch.cat(
                [summary.logLikelihood(g) for summary, g in zip(summaries, gs)]
            )
            assert torch.all((ll - _l).abs() < 0.0001)

        return ll


class RecognitionModel(nn.Module):
    """
    Defines the full-stack RecognitionModel used for neurally-guided search.
    It contains:
        self.featureExtractor: a feed-forward encoder used to encode the I/O examples of a given task.
        This should produce an [n_task x featureExtractor.outputDimensionality] feature encoding tensor.

        self.language_encoder: a feed-forward encoder used to encode the language annotations of a given task.
        This should produce an [n_task x language_encoder.outputDimensionality] language encoding tensor.

        These are both concatenated to produce a task encoding tensor.

        self._MLP: a 2-layer perception module that takes the task encoding tensor and passes it through a hidden layer and an activation layer to produce an [n_task x self.outputDimensionality] tensor.

        self.grammarBuilder: this takes an [n_task x self.outputDimensionality] tensor and produces a transition matrix over the grammar (if contextual); or weights over the unigrams.
    """

    def __init__(
        self,
        example_encoder : ModelLoader | None = None,
        language_encoder:  nn.Module | None = None,
        grammar: Grammar | None = None,
        hidden: list =[64],
        activation: str = "tanh",
        rank: int|None = None,
        contextual: bool =False,
        mask: bool = False,
        cuda: bool = False,
        pretrained_model: RecognitionModel|None = None,
        nearest_encoder: RecognitionModel|None = None,
        nearest_tasks: list|None =None,
        helmholtz_nearest_language: int = 0,
        helmholtz_translations: bool = False,
        id=0):
        """
        _summary_

        Args:
            example_encoder (ModelLoader, optional): Feature Extractor for extracting features from tasks, usually defined in domain files (such as Re2FeatureExamplesEncoder). Defaults to None.

            language_encoder (nn.Module, optional): Language feature extractor (NgramFeaturizer or TokenRecurrentFeatureExtractor as per dreamcoder.py). Defaults to None.
            
            grammar (Grammar, optional): Grammar that the recognition model is trained on. Defaults to None.
            
            hidden (list, optional): List of number of neurons of each hidden layer of self._MLP neural recognition model. Defaults to [64].
            
            activation (str, optional): Activation function of self._MLP neural 
            recognition model. Can take values "sigmoid" | "tanh" | "relu", else throws an error. Defaults to "tanh".
            
            rank (int, optional): Rank of Low-Rank Contextual Grammar Network. Defaults to None (initialized to 16 in the actual model if rank=None and a Low-Rank Contextual Grammar Network is constructed).
            
            contextual (bool, optional): If true, a Contextual Grammar Network is constructed rather than a Grammar Network object. Defaults to False.
            
            mask (bool, optional): If true, a Contextual Grammar Network is constructed for a bigram model with the unconditional bigram transitions is constructed. Defaults to False.
            
            cuda (bool, optional): If true, cuda tensors are used instead of regular tensors. Defaults to False.
            
            pretrained_model (RecognitionModel, optional): Another, previously trained recognition model (may be from a training session that was interrupted previously). Defaults to None.
            
            nearest_encoder (RecognitionModel, optional): Experimental unreleased feature as per dreamcoder.py. Defaults to None.
            
            nearest_tasks (list, optional): List of tasks. Defaults to None.

            helmholtz_nearest_language (int, optional): Label of nearest Helmholtz language. Defaults to 0.
            
            helmholtz_translations (bool, optional): If true, updates the language encoder with language for the Helmholtz entries. Defaults to False.
            
            id (int, optional): Id of recognition model if ensemble of recognition models is being trained. Defaults to 0.

        Raises:
            Exception: If activation function is not "sigmoid" | "tanh" | "relu".
        """
        super(RecognitionModel, self).__init__()
        self.id = id
        self.trained = False
        self.use_cuda = cuda

        self.featureExtractor = example_encoder
        self.language_encoder = language_encoder

        # Use the language encoder to get Helmholtz translations.
        self.helmholtz_translations = helmholtz_translations
        # Encode tasks to lookup nearest language for Helmholtz
        self.encoded_tasks = None  # n_tasks x n_features
        self.nearest_tasks = None  # Sorted list of tasks.
        self.nearest_encoder = None
        self.helmholtz_nearest_language = helmholtz_nearest_language
        if self.helmholtz_nearest_language > 0:
            self.init_helmholtz_nearest_language(
                nearest_encoder=nearest_encoder, nearest_tasks=nearest_tasks
            )
        self.fresh_helmholtz_name = 0  # Fresh names for all Helmholtz entries.

        # Sanity check - make sure that all of the parameters of the
        # feature extractor were added to our parameters as well
        self.feature_dimensions = 0
        if self.featureExtractor is not None:
            if hasattr(example_encoder, "parameters"):
                for parameter in example_encoder.parameters():
                    assert any(
                        myParameter is parameter for myParameter in self.parameters()
                    )
            self.feature_dimensions += self.featureExtractor.outputDimensionality
        if self.language_encoder is not None:
            if hasattr(language_encoder, "parameters"):
                for parameter in language_encoder.parameters():
                    assert any(
                        myParameter is parameter for myParameter in self.parameters()
                    )
            self.feature_dimensions += self.language_encoder.outputDimensionality

        if pretrained_model is not None:
            # Initialize the example encoder and freeze its weights.
            self.featureExtractor.load_state_dict(
                pretrained_model.featureExtractor.state_dict()
            )
            for param in self.featureExtractor.parameters():
                param.requires_grad = False
            # Note that we do *not* reuse the pretrained MLP.

        # Build the multilayer perceptron that is sandwiched between the feature extractor and the grammar
        if activation == "sigmoid":
            activation = nn.Sigmoid
        elif activation == "relu":
            activation = nn.ReLU
        elif activation == "tanh":
            activation = nn.Tanh
        else:
            raise Exception("Unknown activation function " + str(activation))

        self._MLP = nn.Sequential(
            *[
                layer
                for j in range(len(hidden))
                for layer in [
                    nn.Linear(([self.feature_dimensions] + hidden)[j], hidden[j]),
                    activation(),
                ]
            ]
        )

        self.entropy = Entropy()

        if len(hidden) > 0:
            self.outputDimensionality = self._MLP[-2].out_features
            assert self.outputDimensionality == hidden[-1]
        else:
            self.outputDimensionality = self.feature_dimensions

        self.contextual = contextual
        if self.contextual:
            if mask:
                self.grammarBuilder = ContextualGrammarNetwork_Mask(
                    self.outputDimensionality, grammar
                )
            else:
                self.grammarBuilder = ContextualGrammarNetwork_LowRank(
                    self.outputDimensionality, grammar, rank
                )
        else:
            self.grammarBuilder = GrammarNetwork(self.outputDimensionality, grammar)

        self.grammar = ContextualGrammar.fromGrammar(grammar) if contextual else grammar
        self.generativeModel = grammar

        self._auxiliaryPrediction = nn.Linear(
            self.feature_dimensions, len(self.grammar.primitives)
        )
        self._auxiliaryLoss = nn.BCEWithLogitsLoss()

        if cuda:
            self.cuda()

    def get_fresh_helmholtz_name(self):
        """
        Provides fresh names for each Helmholtz entry.

        Returns:
            int: Fresh name for Helmholtz entry.
        """
        self.fresh_helmholtz_name += 1
        return self.fresh_helmholtz_name

    def init_helmholtz_nearest_language(self, nearest_encoder:RecognitionModel, nearest_tasks:list):
        """
        Encode training tasks for kNN featurization of Helmholtz.
        
        Args:
            nearest_encoder (RecognitionModel): Recognition model used for nearest neighbor lookup.
            nearest_tasks (list): List of tasks.
        """
        if self.helmholtz_nearest_language > 1:
            print("Unimplemented: more than 1 nearest neighbor.")
            assert False
        eprint(
            f"Using n={self.helmholtz_nearest_language} nearest language labels to label Helmholtz."
        )
        self.nearest_tasks = sorted(nearest_tasks, key=lambda t: t.name)
        # Naively just attempt to encode all of the tasks at once.
        self.encoded_tasks = nearest_encoder.encode_features_batch_for_lookup(
            self.nearest_tasks
        )
        self.nearest_encoder = nearest_encoder
        self.nearest_encoder.requires_grad = False

    def auxiliaryLoss(self, frontier: Frontier, features: torch.Tensor):
        """
        Calculates the auxiliary loss for the recognition model.

        Args:
            frontier (Frontier): Frontier of programs
            features (torch.Tensor): A self.feature_dimensions dimensional tensor input for the recognition model neural network.

        Returns:
            torch.Tensor: auxiliary loss for the recognition model.
        """
        # Compute a vector of uses
        ls = frontier.bestPosterior.program

        def uses(summary):
            if hasattr(summary, "uses"):
                return torch.tensor(
                    [
                        float(int(p in summary.uses))
                        for p in self.generativeModel.primitives
                    ]
                )
            assert hasattr(summary, "noParent")
            u = uses(summary.noParent) + uses(summary.variableParent)
            for ss in summary.library.values():
                for s in ss:
                    u += uses(s)
            return u

        u = uses(ls)
        u[u > 1.0] = 1.0
        if self.use_cuda:
            u = u.cuda()
        al = self._auxiliaryLoss(self._auxiliaryPrediction(features), u)
        return al

    def taskEmbeddings(self, tasks:list):
        """
        Return taks embeddings of tasks

        Args:
            tasks (list): A list of tasks.

        Returns:
            dict: A dictionary containing task embeddings of tasks.
        """
        return {task: self.encode_features(task).data.cpu().numpy() for task in tasks}

    def encode_features_batch_for_lookup(self, tasks:list):
        """
        Encodes sorted batch of tasks, returns n_tasks x n_encoding_dim tensor.
        
        Args:
            tasks (list): A list of tasks.

        Returns:
            torch.Tensor: A n_tasks x n_encoding_dim tensor.
        """
        print(f"Encoding batch of n={len(tasks)} tasks for lookup only.")
        start = time.time()
        # Naive implementation: encoding them all one by one.
        encoded = torch.stack(
            [self.encode_features(task).detach() for task in tasks], dim=0
        )
        eprint(
            f"Finished encoding batch of n={len(tasks)} tasks in {time.time() - start} seconds."
        )
        return encoded

    def encode_features(self, task:Task):
        """
        Forwards task through the feature layers and concatenates the outputs.

        Args:
            task (Task): A task.

        Returns:
            torch.Tensor: A tensor containing the concatenated outputs of the feature layers.
        """
        features = []
        if self.featureExtractor is not None:
            example_features = self.featureExtractor.featuresOfTask(task)
            if example_features is not None:
                features += [example_features]
        if self.language_encoder is not None:
            language_features = self.language_encoder.featuresOfTask(task)
            if language_features is not None:
                features += [language_features]
        if len(features) < 1:
            return None
        concatenated = torch.cat(features)
        return concatenated

    def forward(self, features: torch.Tensor):
        """
        Returns either a Grammar or a ContextualGrammar.

        Takes as input the concatenation of all of its feature extractor features.
        
        Args:
            features (torch.Tensor): A tensor containing the concatenated feature extractor features.
        
        Returns:
            Grammar | ContextualGrammar: A Grammar or ContextualGrammar object.

        """
        features = self._MLP(features)
        return self.grammarBuilder(features)

    def auxiliaryPrimitiveEmbeddings(self):
        """
        Returns the actual outputDimensionality weight vectors for each of the primitives.
        
        Returns:
            dict: A dictionary containing the actual outputDimensionality weight vectors for each of the primitives.
        """
        auxiliaryWeights = self._auxiliaryPrediction.weight.data.cpu().numpy()
        primitivesDict = {
            self.grammar.primitives[i]: auxiliaryWeights[i, :]
            for i in range(len(self.grammar.primitives))
        }
        return primitivesDict

    def grammarOfTask(self, task:Task):
        """
        Generate a grammar or contextual grammar object for a given task.

        Args:
            task (Task): A task to generate features for

        Returns:
            Grammar | ContextualGrammar: A Grammar or ContextualGrammar object derived by calling forward on the encoded features of the task.
        """

        features = self.encode_features(task)
        if features is None:
            return None
        return self(features)

    def grammarLogProductionsOfTask(self, task:Task):
        """
        Returns the grammar logits from non-contextual models.
        
        Args:
            task (Task): A task.

        Returns:
            torch.Tensor: A tensor containing the grammar logits from non-contextual models, generating by calling the forward() function on the self.grammarBuilder Grammar Network's logProduction neural network .
        """
        features = self.encode_features(task)
        if features is None:
            return None

        if hasattr(self, "hiddenLayers"):
            # Backward compatability with old checkpoints.
            for layer in self.hiddenLayers:
                features = self.activation(layer(features))
            # return features
            return self.noParent[1](features)
        else:
            features = self._MLP(features)

        if self.contextual:
            if hasattr(self.grammarBuilder, "variableParent"):
                return self.grammarBuilder.variableParent.logProductions(features)
            elif hasattr(self.grammarBuilder, "network"):
                return self.grammarBuilder.network(features).view(-1)
            elif hasattr(self.grammarBuilder, "transitionMatrix"):
                return self.grammarBuilder.transitionMatrix(features).view(-1)
            else:
                assert False
        else:
            return self.grammarBuilder.logProductions(features)

    def grammarFeatureLogProductionsOfTask(self, task:Task):
        """
        Extracts the feature vectors from the log productions of a task.

        Args:
            task (Task): A task.

        Returns:
            torch.Tensor: A tensor containing the feature vectors from the log productions of a task.
        """
        return torch.tensor(self.grammarOfTask(task).untorch().featureVector())

    def grammarLogProductionDistanceToTask(self, task:Task, tasks:list):
        """
        Returns the cosine similarity of all other tasks to a given task.
        
        Args:
            task (Task): A task.
            tasks (list): A list of tasks.
        
        Returns:
            np.array: A numpy array containing the cosine similarity of all other tasks to a given task.
        """
        taskLogits = self.grammarLogProductionsOfTask(task).unsqueeze(
            0
        )  # Change to [1, D]
        assert (
            taskLogits is not None
        ), "Grammar log productions are not defined for this task."
        otherTasks = [t for t in tasks if t is not task]  # [nTasks -1 , D]

        # Build matrix of all other tasks.
        otherLogits = torch.stack(
            [self.grammarLogProductionsOfTask(t) for t in otherTasks]
        )
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cosMatrix = cos(taskLogits, otherLogits)
        return cosMatrix.data.cpu().numpy()

    def grammarEntropyOfTask(self, task:Task):
        """
        Returns the entropy of the grammar distribution from non-contextual models for a task.
        
        Args:
            task (Task): A task.
        
        Returns:
            torch.Tensor: A tensor containing the entropy of the grammar distribution from non-contextual models for a task.
        """
        grammarLogProductionsOfTask = self.grammarLogProductionsOfTask(task)

        if grammarLogProductionsOfTask is None:
            return None

        if hasattr(self, "entropy"):
            return self.entropy(grammarLogProductionsOfTask)
        else:
            e = Entropy()
            return e(grammarLogProductionsOfTask)

    def taskAuxiliaryLossLayer(self, tasks:list):
        """
        Returns the auxiliary prediction for a given task.

        Args:
            tasks (list): A list of tasks.

        Returns:
            dict: A dictionary containing the auxiliary loss for each task.
        """
        return {
            task: self._auxiliaryPrediction(self.encode_features(task))
            .view(-1)
            .data.cpu()
            .numpy()
            for task in tasks
        }

    def taskGrammarFeatureLogProductions(self, tasks:list):
        """
        Returns the feature vectors from the log productions of a task.

        Args:
            tasks (list): A list of tasks.

        Returns:
            dict: A dictionary containing the feature vectors from the log productions of a task.
        """
        return {
            task: self.grammarFeatureLogProductionsOfTask(task).data.cpu().numpy()
            for task in tasks
        }

    def taskGrammarLogProductions(self, tasks:list):
        """
        Returns the grammar logits from non-contextual models for a task.

        Args:
            tasks (list): A list of tasks.

        Returns:
            dict: A dictionary containing the grammar logits from non-contextual models for a task.
        """
        return {
            task: self.grammarLogProductionsOfTask(task).data.cpu().numpy()
            for task in tasks
        }

    def taskGrammarStartProductions(self, tasks:list):
        """
        Returns the start productions of a task.

        Args:
            tasks (list): A list of tasks.

        Returns:
            dict: A dictionary containing the start productions of a task.
        """
        return {
            task: np.array([l for l, _1, _2 in g.productions])
            for task in tasks
            for g in [self.grammarOfTask(task).untorch().noParent]
        }

    def taskHiddenStates(self, tasks:list):
        """
        Returns the hidden states of the recognition model for a given task.

        Args:
            tasks (list): A list of tasks.

        Returns:
            dict: A dictionary containing the hidden states of the recognition model for a given task.
        """
        return {
            task: self._MLP(self.encode_features(task)).view(-1).data.cpu().numpy()
            for task in tasks
        }

    def taskGrammarEntropies(self, tasks:list):
        """
        Returns the entropy of the grammar distribution from non-contextual models for a task.

        Args:
            tasks (list): _description_

        Returns:
            dict: A dictionary containing the entropy of the grammar distribution from non-contextual models for a task.
        """
        return {
            task: self.grammarEntropyOfTask(task).data.cpu().numpy() for task in tasks
        }

    def frontierKL(self, frontier:Frontier, auxiliary:bool=False, vectorized:bool=True):
        """
        Returns the KL divergence of the frontier.

        Args:
            frontier (Frontier): A frontier of programs
            auxiliary (bool, optional): If true, uses the original encoded features of the task, else uses a detached tensor as feature when computing auxiliary loss. Defaults to False.
            vectorized (bool, optional): Vectorizes log likelihoods if true. Defaults to True.

        Returns:
            torch.Tensor, torch.Tensor: returns log likelihood and auxiliary loss of frontier.
        """
        features = self.encode_features(frontier.task)
        if features is None:
            return None, None
        # Monte Carlo estimate: draw a sample from the frontier
        entry = frontier.sample()

        al = self.auxiliaryLoss(frontier, features if auxiliary else features.detach())

        if not vectorized:
            g = self(features)
            return -entry.program.logLikelihood(g), al
        else:
            features = self._MLP(features).expand(1, features.size(-1))
            ll = self.grammarBuilder.batchedLogLikelihoods(
                features, [entry.program]
            ).view(-1)
            return -ll, al

    def frontierBiasOptimal(self, frontier:Frontier, auxiliary:bool=False, vectorized:bool=True):
        """
        Returns the bias optimal of the frontier.

        Args:
            frontier (Frontier): _description_
            auxiliary (bool, optional): _description_. Defaults to False.
            vectorized (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if not vectorized:
            features = self.encode_features(frontier.task)
            if features is None:
                return None, None
            al = self.auxiliaryLoss(
                frontier, features if auxiliary else features.detach()
            )
            g = self(features)
            summaries = [entry.program for entry in frontier]
            likelihoods = torch.cat(
                [
                    entry.program.logLikelihood(g) + entry.logLikelihood
                    for entry in frontier
                ]
            )
            best = likelihoods.max()
            return -best, al

        batchSize = len(frontier.entries)
        features = self.encode_features(frontier.task)
        if features is None:
            return None, None
        al = self.auxiliaryLoss(frontier, features if auxiliary else features.detach())
        features = self._MLP(features)
        features = features.expand(batchSize, features.size(-1))  # TODO
        lls = self.grammarBuilder.batchedLogLikelihoods(
            features, [entry.program for entry in frontier]
        )
        actual_ll = torch.Tensor([entry.logLikelihood for entry in frontier])
        lls = lls + (actual_ll.cuda() if self.use_cuda else actual_ll)
        ml = -lls.max()  # Beware that inputs to max change output type
        return ml, al

    def replaceProgramsWithLikelihoodSummaries(self, frontier: Frontier):
        """
        Replaces programs in a frontier with updated programs with accurate likelihood summaries and log priors.

        Args:
            frontier (Frontier): A frontier of programs
        
        Returns:
            Frontier: A frontier of programs with updated likelihood summaries and logPriors.
            
        """

        def make_entry(e):
            if e.tokens is None:
                e.tokens = e.program.left_order_tokens(show_vars=False)
            try:
                return FrontierEntry(
                    program=self.grammar.closedLikelihoodSummary(
                        frontier.task.request, e.program
                    ),
                    logLikelihood=e.logLikelihood,
                    logPrior=self.grammar.logLikelihood(
                        frontier.task.request, e.program
                    ),
                    tokens=e.tokens,
                    test=e.program,
                )
            except:
                return None

        frontier_summaries = [make_entry(e) for e in frontier]
        frontier_summaries = [s for s in frontier_summaries if s is not None]
        if len(frontier_summaries) == 0:
            return None
        else:
            return Frontier(frontier_summaries, task=frontier.task)

    def pairwise_cosine_similarity(self, a: torch.Tensor, b:torch.Tensor, eps: float =1e-8):
        """
        Added eps for numerical stability

        Args:
            a (torch.Tensor): A tensor
            b (torch.Tensor): A tensor
            eps (float, optional): A small constant for numerical stability. Defaults to 1e-8.

        Returns:
            torch.Tensor: A tensor containing the pairwise cosine similarity of a and b.
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def update_helmholtz_language(self, helmholtz_entries:list):
        """
        If self.helmholtz_translations:
            Updates the language encoder with language for the Helmholtz entries.
        If self.helmholtz_nearest_language:
            Updates nearest_name attribute on Helmholtz entries with name
            of nearest training task.
        """
        if self.helmholtz_translations:
            non_empty_helmholtz_entries = [
                e for e in helmholtz_entries if e.task is not None
            ]
            self.language_encoder.update_with_tokenized_helmholtz(
                non_empty_helmholtz_entries, self.generativeModel
            )
            return
        elif self.helmholtz_nearest_language < 1:
            return
        elif len(self.encoded_tasks) < 0:
            return
        else:
            non_empty_helmholtz_entries = [
                e for e in helmholtz_entries if e.task is not None
            ]
            helmholtz_tasks = [e.task for e in non_empty_helmholtz_entries]
            eprint(
                f"Updating language with nearest neighbors for {len(helmholtz_tasks)}/{len(helmholtz_entries)} Helmholtz entries."
            )
            encoded_helmholtz = self.nearest_encoder.encode_features_batch_for_lookup(
                helmholtz_tasks
            )
            # Find nearest between encoded helmholtz and encoded tasks.
            sim_matrix = self.pairwise_cosine_similarity(
                encoded_helmholtz, self.encoded_tasks
            )
            max_similarity = torch.argmax(sim_matrix, 1)
            # Update names of all the helmholtz tasks
            for ind, e in enumerate(non_empty_helmholtz_entries):
                nearest_ind = max_similarity[ind]
                e.task.nearest_name = self.nearest_tasks[nearest_ind].name

    def train(
        self,
        frontiers: list,
        _=None,
        steps: int|None = None,
        lr: float = 0.001,
        topK: int = 5,
        CPUs: int = 1,
        timeout: int|None = None,
        evaluationTimeout: float =0.001,
        helmholtzFrontiers: list =[],
        helmholtzRatio: float = 0.0,
        helmholtzBatch: int = 500,
        biasOptimal: int|None = None,
        defaultRequest: Task.request|None = None,
        auxLoss: bool = False,
        vectorized:bool = True,
        epochs:int|None = None,
        generateNewHelmholtz: bool =True,
    ):
        """
        Trains the recognition model.

        Args:
            frontiers(list): list of frontiers of programs to train on.

            steps (int): Number of gradient steps to take.

            lr (float): Learning rate for the optimizer.

            topK (int): Number of top programs to consider.

            CPUs (int): Number of CPUs to use for parallel processing.

            timeout (int): Timeout for training.

            evaluationTimeout (int): Timeout for evaluation.

            helmholtzFrontiers (list): Frontiers from programs enumerated from generative model (optional). If helmholtzFrontiers is not provided then we will sample programs during training

            helmholtzRatio (float): What fraction of the training data should be forward samples from the generative model?

            helmholtzBatch (int): Number of programs to sample from the generative model.

            biasOptimal (int): If true, uses bias optimal training.

            defaultRequest (Task.request): Default request for the task (i.e. task type) if frontiers are empty.

            auxLoss (bool): If true, uses auxiliary loss.

            vectorized (bool): If true, uses vectorized log likelihoods.

            epochs (int): Number of epochs to train for.

            generateNewHelmholtz (bool): If true, generates new Helmholtz entries during training.
        """
        assert (
            (steps is not None) or (timeout is not None) or (epochs is not None)
        ), "Cannot train recognition model without either a bound on the number of gradient steps, bound on the training time, or number of epochs"
        if steps is None:
            steps = 9999999
        if epochs is None:
            epochs = 9999999
        if timeout is None:
            timeout = 9999999
        if biasOptimal is None:
            biasOptimal = len(helmholtzFrontiers) > 0

        requests = [frontier.task.request for frontier in frontiers]
        if len(requests) == 0 and helmholtzRatio > 0 and len(helmholtzFrontiers) == 0:
            assert (
                defaultRequest is not None
            ), "You are trying to random Helmholtz training, but don't have any frontiers. Therefore we would not know the type of the program to sample. Try specifying defaultRequest=..."
            requests = [defaultRequest]
        frontiers = [
            frontier.topK(topK).normalize()
            for frontier in frontiers
            if not frontier.empty
        ]

        if self.featureExtractor is None:
            eprint("No feature extractor; no Helmholtz.")
            helmholtzRatio = 0.0
        elif len(frontiers) == 0:
            eprint(
                "You didn't give me any nonempty replay frontiers to learn from. Going to learn from 100% Helmholtz samples"
            )
            helmholtzRatio = 1.0

        # This determines whether we have pre-enumerated a set of helmholtzFrontiers: if we have not,
        # we will need to sample them during the training loop itself.
        randomHelmholtz = len(helmholtzFrontiers) < 1
        if randomHelmholtz:
            print(
                "No pre-enumerated helmholtzFrontiers: we will sample randomly during the training loop."
            )
            helmholtzFrontiers = []

        class HelmholtzEntry:
            """Wrapper class to mix executed Helmholtz programs as 'tasks' into the training schedule."""

            def __init__(self, frontier, owner):
                frontier.task.name += f"{owner.get_fresh_helmholtz_name()}"
                self.request = frontier.task.request
                self.task = None
                self.programs = [e.program for e in frontier]
                self.program_tokens = [e.tokens for e in frontier]
                self.frontier = Thunk(
                    lambda: owner.replaceProgramsWithLikelihoodSummaries(frontier)
                )
                self.owner = owner

            def clear(self):
                self.task = None

            def calculateTask(self):
                assert self.task is None
                p = random.choice(self.programs)
                task = self.owner.featureExtractor.taskOfProgram(p, self.request)
                if task is not None:
                    task.name += f"{self.owner.get_fresh_helmholtz_name()}"
                return task

            def makeFrontier(self):
                assert self.task is not None
                f = Frontier(self.frontier.force().entries, task=self.task)
                return f

            def setTask(self, task):
                assert self.task is None
                if task is None:
                    self.task = None
                    return
                task.name += f"{self.owner.get_fresh_helmholtz_name()}"
                self.task = task

        # Should we recompute tasks on the fly from Helmholtz?  This
        # should be done if the task is stochastic, or if there are
        # different kinds of inputs on which it could be run. For
        # example, lists and strings need this; towers and graphics do
        # not. There is no harm in recomputed the tasks, it just
        # wastes time.
        if not hasattr(self.featureExtractor, "recomputeTasks") and (
            not self.featureExtractor is None
        ):
            self.featureExtractor.recomputeTasks = True

        ## Initializes HelmholtzEntry objects from the pre-enumerated frontiers.
        helmholtzFrontiers = [HelmholtzEntry(f, self) for f in helmholtzFrontiers]
        if len(helmholtzFrontiers) > 0:
            # Generate natural language for the helmholtz programs.
            self.update_helmholtz_language(helmholtzFrontiers)
        random.shuffle(helmholtzFrontiers)

        ## Helper methods for getting Helmholtz entries in the training loop, which can involve
        # sampling new entries if we run out.
        helmholtzIndex = [0]

        def getHelmholtz(max_tries=0):
            """
            Helper method to get the Helmholtz frontiers we have generated, or sample new ones if we ran out.
            
            Args:
                max_tries (int, optional): Number of tries to get Helmholtz frontiers. Defaults to 0.
            
            Returns:
                Frontier: A Frontier of Helmholtz tasks.
            """
            switchToRandom = False
            if max_tries > 100:
                print("Switching to random...")
                switchToRandom = True
            if randomHelmholtz or switchToRandom:
                if helmholtzIndex[0] >= len(helmholtzFrontiers):
                    updateHelmholtzTasks(switchToRandom=switchToRandom)
                    helmholtzIndex[0] = 0
                    return getHelmholtz(max_tries + 1)
                helmholtzIndex[0] += 1
                return helmholtzFrontiers[helmholtzIndex[0] - 1].makeFrontier()

            if helmholtzIndex[0] >= len(helmholtzFrontiers):
                helmholtzIndex[0] = 0
                random.shuffle(helmholtzFrontiers)
                if self.featureExtractor.recomputeTasks:
                    for fp in helmholtzFrontiers:
                        fp.clear()
                    return getHelmholtz(
                        max_tries + 1
                    )  # because we just cleared everything

            f = helmholtzFrontiers[helmholtzIndex[0]]
            if f.task is None:
                with timing("Evaluated another batch of Helmholtz tasks"):
                    updateHelmholtzTasks()
                return getHelmholtz(max_tries + 1)

            helmholtzIndex[0] += 1
            if helmholtzIndex[0] >= len(helmholtzFrontiers):
                helmholtzIndex[0] = 0
                random.shuffle(helmholtzFrontiers)
                if self.featureExtractor.recomputeTasks:
                    for fp in helmholtzFrontiers:
                        fp.clear()
                    return getHelmholtz(
                        max_tries + 1
                    )  # because we just cleared everything
            assert f.task is not None
            return f.makeFrontier()

        def updateHelmholtzTasks(switchToRandom=False):
            """
            Update provided Helmholtz tasks with sampled Helmoltz tasks.

            Args:
                switchToRandom (bool, optional): If true, switches to randomly sampling Helmholtz tasks. Defaults to False.s
            """
            updateCPUs = (
                CPUs
                if hasattr(self.featureExtractor, "parallelTaskOfProgram")
                and self.featureExtractor.parallelTaskOfProgram
                else 1
            )
            # if updateCPUs > 1:
            #     eprint(
            #         "Updating Helmholtz tasks with",
            #         updateCPUs,
            #         "CPUs",
            #         "while using",
            #         getThisMemoryUsage(),
            #         "memory",
            #     )

            if generateNewHelmholtz and randomHelmholtz or switchToRandom:
                print("Sampling new helmholtz frontiers during training.")
                newFrontiers = self.sampleManyHelmholtz(requests, helmholtzBatch, CPUs)
                newEntries = []
                for f in newFrontiers:
                    e = HelmholtzEntry(f, self)
                    e.task = f.task
                    newEntries.append(e)
                helmholtzFrontiers.clear()
                helmholtzFrontiers.extend(newEntries)
                self.update_helmholtz_language(helmholtzFrontiers)
                print(f"We now have {len(helmholtzFrontiers)} Helmholtz entries.")
                return

            # Save some memory by freeing up the tasks as we go through them
            if self.featureExtractor.recomputeTasks:
                for hi in range(
                    max(
                        0,
                        helmholtzIndex[0] - helmholtzBatch,
                        min(helmholtzIndex[0], len(helmholtzFrontiers)),
                    )
                ):
                    helmholtzFrontiers[hi].clear()

            if hasattr(self.featureExtractor, "tasksOfPrograms"):
                eprint("batching task calculation")
                newTasks = self.featureExtractor.tasksOfPrograms(
                    [
                        random.choice(hf.programs)
                        for hf in helmholtzFrontiers[
                            helmholtzIndex[0] : helmholtzIndex[0] + helmholtzBatch
                        ]
                    ],
                    [
                        hf.request
                        for hf in helmholtzFrontiers[
                            helmholtzIndex[0] : helmholtzIndex[0] + helmholtzBatch
                        ]
                    ],
                )
            else:
                newTasks = [
                    hf.calculateTask()
                    for hf in helmholtzFrontiers[
                        helmholtzIndex[0] : helmholtzIndex[0] + helmholtzBatch
                    ]
                ]
                """
                # catwong: Disabled for ensemble training.
                newTasks = \
                           parallelMap(updateCPUs,
                                       lambda f: f.calculateTask(),
                                       helmholtzFrontiers[helmholtzIndex[0]:helmholtzIndex[0] + helmholtzBatch],
                                       seedRandom=True)
                """
            badIndices = []
            endingIndex = min(
                helmholtzIndex[0] + helmholtzBatch, len(helmholtzFrontiers)
            )
            for i in range(helmholtzIndex[0], endingIndex):
                helmholtzFrontiers[i].setTask(newTasks[i - helmholtzIndex[0]])
                if helmholtzFrontiers[i].task is None:
                    badIndices.append(i)
            # Permanently kill anything which failed to give a task
            for i in reversed(badIndices):
                assert helmholtzFrontiers[i].task is None
                del helmholtzFrontiers[i]
            self.update_helmholtz_language(
                helmholtzFrontiers
            )  # Generate language for all sampled programs.

        # We replace each program in the frontier with its likelihoodSummary
        # This is because calculating likelihood summaries requires juggling types
        # And type stuff is expensive!
        initial_num_frontiers = len(frontiers)
        frontiers = [self.replaceProgramsWithLikelihoodSummaries(f) for f in frontiers]
        frontiers = [f for f in frontiers if f is not None]
        frontiers = [f.normalize() for f in frontiers]
        print(
            f"Attempted frontier likelihood normalization. Initial frontiers: {initial_num_frontiers}. Now: {len(frontiers)}"
        )

        feature_extractor_names = [
            str(encoder.__class__.__name__)
            for encoder in (self.featureExtractor, self.language_encoder)
            if encoder is not None
        ]
        eprint(
            "(ID=%d): Training a recognition model from %d frontiers, %d%% Helmholtz"
            % (self.id, len(frontiers), int(helmholtzRatio * 100))
        )
        eprint(f"Feature extractors: [{feature_extractor_names}]")
        eprint(
            "(ID=%d): Got %d Helmholtz frontiers - random Helmholtz training? : %s"
            % (self.id, len(helmholtzFrontiers), len(helmholtzFrontiers) < 2)
        )
        eprint("(ID=%d): Contextual? %s" % (self.id, str(self.contextual)))
        eprint("(ID=%d): Bias optimal? %s" % (self.id, str(biasOptimal)))
        eprint(
            f"(ID={self.id}): Aux loss? {auxLoss} (n.b. we train a 'auxiliary' classifier anyway - this controls if gradients propagate back to the future extractor)"
        )

        # The number of Helmholtz samples that we generate at once
        # Should only affect performance and shouldn't affect anything else
        helmholtzSamples = []

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-3, amsgrad=True)
        start = time.time()
        losses, descriptionLengths, realLosses, dreamLosses, realMDL, dreamMDL = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        classificationLosses = []
        totalGradientSteps = 0
        for i in range(1, epochs + 1):
            if timeout and time.time() - start > timeout:
                break

            if totalGradientSteps > steps:
                break

            if helmholtzRatio < 1.0:
                permutedFrontiers = list(frontiers)
                random.shuffle(permutedFrontiers)
            else:
                permutedFrontiers = [None]

            finishedSteps = False
            for frontier in permutedFrontiers:
                # Randomly decide whether to sample from the generative model
                dreaming = random.random() < helmholtzRatio
                if dreaming:
                    try:
                        frontier = getHelmholtz()
                    except:
                        continue  # Just use the training frontier.
                self.zero_grad()
                loss, classificationLoss = (
                    self.frontierBiasOptimal(
                        frontier, auxiliary=auxLoss, vectorized=vectorized
                    )
                    if biasOptimal
                    else self.frontierKL(
                        frontier, auxiliary=auxLoss, vectorized=vectorized
                    )
                )
                if loss is None:
                    if not dreaming:
                        eprint(
                            "ERROR: Could not extract features during experience replay."
                        )
                        eprint("Task is:", frontier.task)
                        eprint(
                            "Aborting - we need to be able to extract features of every actual task."
                        )
                        assert False
                    else:
                        continue
                if is_torch_invalid(loss):
                    eprint("Invalid real-data loss!")
                else:
                    (loss + classificationLoss).backward()
                    classificationLosses.append(classificationLoss.data.item())
                    optimizer.step()
                    totalGradientSteps += 1
                    losses.append(loss.data.item())
                    descriptionLengths.append(min(-e.logPrior for e in frontier))
                    if dreaming:
                        dreamLosses.append(losses[-1])
                        dreamMDL.append(descriptionLengths[-1])
                    else:
                        realLosses.append(losses[-1])
                        realMDL.append(descriptionLengths[-1])
                    if totalGradientSteps > steps:
                        break  # Stop iterating, then print epoch and loss, then break to finish.

            if (i == 1 or i % 10 == 0) or (totalGradientSteps % 10 == 0) and losses:
                eprint("(ID=%d): " % self.id, "Epoch", i, "Loss", mean(losses))
                if realLosses and dreamLosses:
                    eprint(
                        "(ID=%d): " % self.id,
                        "\t\t(real loss): ",
                        mean(realLosses),
                        "\t(dream loss):",
                        mean(dreamLosses),
                    )
                eprint(
                    "(ID=%d): " % self.id,
                    "\tvs MDL (w/o neural net)",
                    mean(descriptionLengths),
                )
                if realMDL and dreamMDL:
                    eprint(
                        "\t\t(real MDL): ",
                        mean(realMDL),
                        "\t(dream MDL):",
                        mean(dreamMDL),
                    )
                eprint(
                    "(ID=%d): " % self.id,
                    "\t%d cumulative gradient steps. %f steps/sec"
                    % (totalGradientSteps, totalGradientSteps / (time.time() - start)),
                )
                eprint(
                    "(ID=%d): " % self.id,
                    "\t%d-way auxiliary classification loss"
                    % len(self.grammar.primitives),
                    sum(classificationLosses) / len(classificationLosses),
                )
                (
                    losses,
                    descriptionLengths,
                    realLosses,
                    dreamLosses,
                    realMDL,
                    dreamMDL,
                ) = ([], [], [], [], [], [])
                classificationLosses = []
                gc.collect()

        eprint(
            "(ID=%d): " % self.id,
            " Trained recognition model in",
            time.time() - start,
            "seconds",
        )
        self.trained = True
        return self

    def sampleHelmholtz(self, requests:list, statusUpdate = None, seed:int|None =None):
        """
        Samples a program and derives a Helmholtz task from the prior.
        
        Args:
            requests (list): A list of requests.
            statusUpdate (None, optional): Status update on Helmholtz sampling which triggers a flush. Defaults to None.
            seed (int, optional): Seed for randomly sampling a program. Defaults to None.

        Returns:
            Frontier: A frontier containing the randomly sampled program and the corresponding task.
        """
        if seed is not None:
            random.seed(seed)
        request = random.choice(requests)

        program = self.generativeModel.sample(request, maximumDepth=6, maxAttempts=100)
        if program is None:
            return None
        task = self.featureExtractor.taskOfProgram(program, request)

        if statusUpdate is not None:
            flushEverything()
        if task is None:
            return None

        if hasattr(self.featureExtractor, "lexicon"):
            if hasattr(self.featureExtractor, "useTask"):
                if self.featureExtractor.tokenize(task) is None:
                    return None
            else:
                if self.featureExtractor.tokenize(task.examples) is None:
                    return None

        ll = self.generativeModel.logLikelihood(request, program)
        frontier = Frontier(
            [FrontierEntry(program=program, logLikelihood=0.0, logPrior=ll)], task=task
        )
        return frontier

    def sampleManyHelmholtz(self, requests:list, N:int, CPUs:int):
        """
        Samples many programs from the prior.

        Args:
            requests (list): A list of requests.
            N (int): Number of programs to sample.
            CPUs (int): Number of CPUs to use for parallel processing.

        Returns:
            list: A list of frontiers containing the randomly sampled programs and the corresponding tasks.
        """

        eprint("Sampling %d programs from the prior on %d CPUs..." % (N, CPUs))
        flushEverything()
        frequency = N / 50
        startingSeed = random.random()

        # Sequentially for ensemble training.
        samples = [
            self.sampleHelmholtz(
                requests,
                statusUpdate="." if n % frequency == 0 else None,
                seed=startingSeed + n,
            )
            for n in range(N)
        ]

        # (cathywong) Disabled for ensemble training.
        # samples = parallelMap(
        #     1,
        #     lambda n: self.sampleHelmholtz(requests,
        #                                    statusUpdate='.' if n % frequency == 0 else None,
        #                                    seed=startingSeed + n),
        #     range(N))
        eprint()
        flushEverything()
        samples = [z for z in samples if z is not None]
        eprint()
        eprint("Got %d/%d valid samples." % (len(samples), N))
        flushEverything()

        return samples

    def enumerateFrontiers(
        self,
        tasks:list,
        enumerationTimeout:int|None = None,
        testing:bool = False,
        solver:str|None = None,
        CPUs:int = 1,
        frontierSize:int|None = None,
        maximumFrontier:float|None = None,
        evaluationTimeout:int|None = None,
        max_mem_per_enumeration_thread:int = 1000000,
        solver_directory:str = ".",  # Default solver directory is top level in original DreamCoder.
        likelihood_model:str = INDUCTIVE_EXAMPLES_LIKELIHOOD_MODEL,
    ):
        """
        Enumerates frontiers for tasks.

        Args:
            tasks (list): A list of tasks.
            enumerationTimeout (int | None, optional): Timeour for enumerating programs. Defaults to None.
            testing (bool, optional): If true, evaluation occurs on held-out testing tasks. Defaults to False.
            solver (str | None, optional): Can be one of "ocaml", "pypy", or "python". Defaults to None.
            CPUs (int, optional): Number of CPUs for multicore enumeration. Defaults to 1.
            frontierSize (int | None, optional): Unused parameter denoting size of frontiers. Defaults to None.
            maximumFrontier (float | None, optional): Float from which sum of all log-likelihoods > 0.1 is later subtracted during enumeration to compute dictionary of maximum frontiers. Defaults to None.
            evaluationTimeout (int | None, optional): Timeout for evaluating programs. Defaults to None.
            max_mem_per_enumeration_thread (int, optional): Maximum memory used in each enumeration thread. Defaults to 1000000.
            solver_directory (str, optional): Directory containing all solvers. Defaults to ".".

        Returns:
            list, float: A list of frontiers, and the best search time.
        """
        with timing("Evaluated recognition model"):
            grammars = {task: self.grammarOfTask(task) for task in tasks}
            # untorch seperately to make sure you filter out None grammars
            grammars = {
                task: grammar.untorch()
                for task, grammar in grammars.items()
                if grammar is not None
            }

        return multicoreEnumeration(
            grammars,
            tasks,
            testing=testing,
            solver=solver,
            enumerationTimeout=enumerationTimeout,
            CPUs=CPUs,
            maximumFrontier=maximumFrontier,
            evaluationTimeout=evaluationTimeout,
            unigramGrammar=self.generativeModel,
            max_mem_per_enumeration_thread=max_mem_per_enumeration_thread,
            solver_directory=solver_directory,
            likelihood_model_string=likelihood_model,
        )


class RecurrentFeatureExtractor(nn.Module):
    def __init__(
        self,
        _=None,
        tasks:list =None,
        cuda:bool =False,
        # what are the symbols that can occur in the inputs and
        # outputs
        lexicon: list | None = None,
        # how many hidden units
        H:int =32,
        # Should the recurrent units be bidirectional?
        bidirectional:bool = False,
        # What should be the timeout for trying to construct Helmholtz tasks?
        helmholtzTimeout:float = 0.25,
        # What should be the timeout for running a Helmholtz program?
        helmholtzEvaluationTimeout:float = 0.25,
        special_encoder:bool = False,
    ):
        """
        Recurrent Feature Extractor for language descriptions of tasks.

        Args:
            tasks (list, optional): A list of tasks. Defaults to None.
            cuda (bool, optional): If true, cuda tensors are used. Defaults to False.
            lexicon (list | None, optional): A list of words and symbols present in the language descriptions. Defaults to None.
            H (int, optional): Number of hiddent units in the self.model network. Defaults to 32.
            bidirectional (bool, optional): If true, bidirectional recurrent units are used. Defaults to False.
            helmholtzTimeout (float, optional): Maximum timeout for helmholtz program generation. Defaults to 0.25.
            helmholtzEvaluationTimeout (float, optional): Maximum timeout for helmholtz program evaluation. Defaults to 0.25.
            special_encoder (bool, optional): If true, use special encoder on symbols in the lexicon and adjust embedding dimensions accordingly. Defaults to False.
        """
        super(RecurrentFeatureExtractor, self).__init__()

        assert (
            tasks is not None
        ), "You must provide a list of all of the tasks, both those that have been hit and those that have not been hit. Input examples are sampled from these tasks."

        # maps from a requesting type to all of the inputs that we ever saw with that request
        self.requestToInputs = {
            tp: [list(map(fst, t.examples)) for t in tasks if t.request == tp]
            for tp in {t.request for t in tasks}
        }

        inputTypes = {t for task in tasks for t in task.request.functionArguments()}
        # maps from a type to all of the inputs that we ever saw having that type
        self.argumentsWithType = {
            tp: [
                x
                for t in tasks
                for xs, _ in t.examples
                for tpp, x in zip(t.request.functionArguments(), xs)
                if tpp == tp
            ]
            for tp in inputTypes
        }
        self.requestToNumberOfExamples = {
            tp: [len(t.examples) for t in tasks if t.request == tp]
            for tp in {t.request for t in tasks}
        }
        self.helmholtzTimeout = helmholtzTimeout
        self.helmholtzEvaluationTimeout = helmholtzEvaluationTimeout
        self.parallelTaskOfProgram = True

        assert lexicon
        lexicon = sorted(lexicon)
        self.specialSymbols = [
            "STARTING",  # start of entire sequence
            "ENDING",  # ending of entire sequence
            "STARTOFOUTPUT",  # begins the start of the output
            "ENDOFINPUT",  # delimits the ending of an input - we might have multiple inputs
        ]
        lexicon += self.specialSymbols
        self.lexicon = lexicon
        # Note: 1 indexed!
        self.symbolToIndex = {symbol: index + 1 for index, symbol in enumerate(lexicon)}
        self.indexToSymbol = {index + 1: symbol for index, symbol in enumerate(lexicon)}

        if special_encoder:
            embedding_dim, encoder = self.special_encoder(self.symbolToIndex)

        else:
            embedding_dim = H
            encoder = nn.Embedding(len(lexicon) + 1, H)  # Allow 1 indexed.
        self.encoder = encoder

        self.H = H
        self.bidirectional = bidirectional

        layers = 1

        model = nn.GRU(embedding_dim, H, layers, bidirectional=bidirectional)
        self.model = model

        self.use_cuda = cuda
        self.startingIndex = self.symbolToIndex["STARTING"]
        self.endingIndex = self.symbolToIndex["ENDING"]
        self.startOfOutputIndex = self.symbolToIndex["STARTOFOUTPUT"]
        self.endOfInputIndex = self.symbolToIndex["ENDOFINPUT"]

        # Maximum number of inputs/outputs we will run the recognition
        # model on per task
        # This is an optimization hack
        self.MAXINPUTS = 100

        if cuda:
            self.cuda()

    @property
    def outputDimensionality(self):
        """
        Returns:
            int: Number of hidden units in self.model network
        """
        return self.H

    # modify examples before forward (to turn them into iterables of lexicon)
    # you should override this if needed
    def tokenize(self, x):
        return x

    def symbolEmbeddings(self):
        return {
            s: self.encoder(variable([self.symbolToIndex[s]]))
            .squeeze(0)
            .data.cpu()
            .numpy()
            for s in self.lexicon
            if not (s in self.specialSymbols)
        }

    def packExamples(self, examples:list):
        """
        Args:
            examples (list): List of tuples.
        Returns:
            tuple: Packed encoded examples and sizes.
        """
        """IMPORTANT! xs must be sorted in decreasing order of size because pytorch is stupid"""
        es = []
        sizes = []
        for xs, y in examples:
            e = [self.startingIndex]
            for x in xs:
                for s in x:
                    e.append(self.symbolToIndex[s])
                e.append(self.endOfInputIndex)
            e.append(self.startOfOutputIndex)
            for s in y:
                e.append(self.symbolToIndex[s])
            e.append(self.endingIndex)
            if es != []:
                assert len(e) <= len(
                    es[-1]
                ), "Examples must be sorted in decreasing order of their tokenized size. This should be transparently handled in recognition.py, so if this assertion fails it isn't your fault as a user of EC but instead is a bug inside of EC."
            es.append(e)
            sizes.append(len(e))

        m = max(sizes)
        # padding
        for j, e in enumerate(es):
            es[j] += [self.endingIndex] * (m - len(e))
        x = variable(es, cuda=self.use_cuda)

        x = self.encoder(x)

        # x: (batch size, maximum length, E)
        x = x.permute(1, 0, 2)
        # x: TxBxE
        x = pack_padded_sequence(x, sizes)
        return x, sizes

    def examplesEncoding(self, examples:list):
        """
        Args:
            examples (list): List of examples.

        Returns:
            torch.Tensor: Encoded examples.
        """
        examples = sorted(
            examples,
            key=lambda xs_y: sum(len(z) + 1 for z in xs_y[0]) + len(xs_y[1]),
            reverse=True,
        )
        x, sizes = self.packExamples(examples)
        outputs, hidden = self.model(x)
        # outputs, sizes = pad_packed_sequence(outputs)
        # I don't know whether to return the final output or the final hidden
        # activations...
        return hidden[0, :, :] + hidden[1, :, :]

    def forward(self, examples_or_task):
        """
        Forward pass through the RecurrentFeatureExtractor network.

        Args:
            examples_or_task (_type_): _description_

        Returns:
            torch.Tensor: Averahe activations across all examples.
        """
        # Takes either the examples themselves, or the task, depending on the tokenization function.
        # If the self.useTask == True, this is a task.
        tokenized = self.tokenize(examples_or_task)
        if not tokenized:
            return None

        if hasattr(self, "MAXINPUTS") and len(tokenized) > self.MAXINPUTS:
            tokenized = list(tokenized)
            random.shuffle(tokenized)
            tokenized = tokenized[: self.MAXINPUTS]
        e = self.examplesEncoding(tokenized)
        # max pool
        # e,_ = e.max(dim = 0)

        # take the average activations across all of the examples
        # I think this might be better because we might be testing on data
        # which has far more o far fewer examples then training
        e = e.mean(dim=0)
        return e

    def featuresOfTask(self, t:Task):
        """
        Compute features of a task.

        Args:
            t (Task): A task

        Returns:
            torch.Tensor: Features of the task.
        """
        if hasattr(self, "useFeatures"):
            f = self(t.features)
        elif hasattr(self, "useTask"):
            f = self(t)
        else:
            # Featurize the examples directly.
            f = self(t.examples)
        return f

    def taskOfProgram(self, p: Program, tp: tuple):
        """
        Generate a task from a program.

        Args:
            p (Program): A program.
            tp (tuple): A tuple of and input, output pair.

        Returns:
            Task: A task.
        """
        # TODO -- remove this
        self.helmholtzTimeout, self.helmholtzEvaluationTimeout = 0.25, 0.25
        # half of the time we randomly mix together inputs
        # this gives better generalization on held out tasks
        # the other half of the time we train on sets of inputs in the training data
        # this gives better generalization on unsolved training tasks
        def is_not_degenerate_outputs(examples):
            """Ensure that we don't have all degenerate outputs, in which every example is the same."""
            outputs = [y for (xs, y) in examples]
            return not (all(y == outputs[0] for y in outputs))

        if random.random() < 0.5:

            def randomInput(t):
                return random.choice(self.argumentsWithType[t])

            # Loop over the inputs in a random order and pick the first ones that
            # doesn't generate an exception

            startTime = time.time()
            examples = []

            while True:
                # TIMEOUT! this must not be a very good program
                if time.time() - startTime > self.helmholtzTimeout:
                    return None

                # Grab some random inputs
                xs = [randomInput(t) for t in tp.functionArguments()]
                try:
                    y = runWithTimeout(
                        lambda: p.runWithArguments(xs), self.helmholtzEvaluationTimeout
                    )
                    examples.append((tuple(xs), y))
                    if len(examples) >= random.choice(
                        self.requestToNumberOfExamples[tp]
                    ):
                        if is_not_degenerate_outputs(examples):
                            return Task("Helmholtz", tp, examples)
                        return None
                except Exception as e:
                    continue  # Try searching for more inputs on which we can run.

        else:
            candidateInputs = list(self.requestToInputs[tp])
            random.shuffle(candidateInputs)
            for xss in candidateInputs:
                ys = []
                for xs in xss:
                    try:
                        y = runWithTimeout(
                            lambda: p.runWithArguments(xs),
                            self.helmholtzEvaluationTimeout,
                        )
                    except Exception as e:
                        return None
                    ys.append(y)
                if len(ys) == len(xss):
                    examples = list(zip(xss, ys))
                    if is_not_degenerate_outputs(examples):
                        return Task("Helmholtz", tp, examples)
                    return None
            return None


class LowRank(nn.Module):
    """
    Module that outputs a rank R matrix of size m by n from input of size i.
    """

    def __init__(self, i:int, m:int, n:int, r:int):
        """
        Args:
            i: input dimension
            m: output rows
            n: output columns
            r: maximum rank. if this is None then the output will be full-rank
        """
        super(LowRank, self).__init__()

        self.m = m
        self.n = n

        maximumPossibleRank = min(m, n)
        if r is None:
            r = maximumPossibleRank

        if r < maximumPossibleRank:
            self.factored = True
            self.A = nn.Linear(i, m * r)
            self.B = nn.Linear(i, n * r)
            self.r = r
        else:
            self.factored = False
            self.M = nn.Linear(i, m * n)

    def forward(self, x:torch.Tensor):
        """
        Forward pass through the LowRank network.        

        Args:
            x (torch.Tensor): Input tensor for neural network.

        Returns:
            torch.Tensor: Output tensor from low-rank neural network.
        """
        sz = x.size()
        if len(sz) == 1:
            B = 1
            x = x.unsqueeze(0)
            needToSqueeze = True
        elif len(sz) == 2:
            B = sz[0]
            needToSqueeze = False
        else:
            assert (
                False
            ), "LowRank expects either a 1-dimensional tensor or a 2-dimensional tensor"

        if self.factored:
            a = self.A(x).view(B, self.m, self.r)
            b = self.B(x).view(B, self.r, self.n)
            y = a @ b
        else:
            y = self.M(x).view(B, self.m, self.n)
        if needToSqueeze:
            y = y.squeeze(0)
        return y


class DummyFeatureExtractor(nn.Module):
    def __init__(self, tasks, testingTasks=[], cuda=False):
        super(DummyFeatureExtractor, self).__init__()
        self.outputDimensionality = 1
        self.recomputeTasks = False

    def featuresOfTask(self, t):
        return variable([0.0]).float()

    def featuresOfTasks(self, ts):
        return variable([[0.0]] * len(ts)).float()

    def taskOfProgram(self, p, t):
        return Task("dummy task", t, [])


class RandomFeatureExtractor(nn.Module):
    def __init__(self, tasks):
        super(RandomFeatureExtractor, self).__init__()
        self.outputDimensionality = 1
        self.recomputeTasks = False

    def featuresOfTask(self, t):
        return variable([random.random()]).float()

    def featuresOfTasks(self, ts):
        return variable([[random.random()] for _ in range(len(ts))]).float()

    def taskOfProgram(self, p, t):
        return Task("dummy task", t, [])


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ImageFeatureExtractor(nn.Module):
    def __init__(self, inputImageDimension, resizedDimension=None, channels=1):
        super(ImageFeatureExtractor, self).__init__()

        self.resizedDimension = resizedDimension or inputImageDimension
        self.inputImageDimension = inputImageDimension
        self.channels = channels

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        # channels for hidden
        hid_dim = 64
        z_dim = 64

        self.encoder = nn.Sequential(
            conv_block(channels, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten(),
        )

        # Each layer of the encoder halves the dimension, except for the last layer which flattens
        outputImageDimensionality = self.resizedDimension / (
            2 ** (len(self.encoder) - 1)
        )
        self.outputDimensionality = int(
            z_dim * outputImageDimensionality * outputImageDimensionality
        )

    def forward(self, v):
        """1 channel: v: BxWxW or v:WxW
        > 1 channel: v: BxCxWxW or v:CxWxW"""

        insertBatch = False
        variabled = variable(v).float()
        if self.channels == 1:  # insert channel dimension
            if len(variabled.shape) == 3:  # batching
                variabled = variabled[:, None, :, :]
            elif len(variabled.shape) == 2:  # no batching
                variabled = variabled[None, :, :]
                insertBatch = True
            else:
                assert False
        else:  # expect to have a channel dimension
            if len(variabled.shape) == 4:
                pass
            elif len(variabled.shape) == 3:
                insertBatch = True
            else:
                assert False

        if insertBatch:
            variabled = torch.unsqueeze(variabled, 0)

        y = self.encoder(variabled)
        if insertBatch:
            y = y[0, :]
        return y


class JSONFeatureExtractor(object):
    def __init__(self, tasks, cudaFalse):
        # self.averages, self.deviations = Task.featureMeanAndStandardDeviation(tasks)
        # self.outputDimensionality = len(self.averages)
        self.cuda = cuda
        self.tasks = tasks

    def stringify(self, x):
        # No whitespace #maybe kill the seperators
        return json.dumps(x, separators=(",", ":"))

    def featuresOfTask(self, t):
        # >>> t.request to get the type
        # >>> t.examples to get input/output examples
        # this might actually be okay, because the input should just be nothing
        # return [(self.stringify(inputs), self.stringify(output))
        #        for (inputs, output) in t.examples]
        return [(list(output),) for (inputs, output) in t.examples]
