from dreamcoder.likelihoodModel import AllOrNothingLikelihoodModel
from dreamcoder.grammar import *
from dreamcoder.utilities import get_root_dir, limit_virtual_memory_fn
import dreamcoder.neurosym as ns
from dreamcoder.neurosym.dsl.abstraction import _with_index_parameters
from dreamcoder.neurosym.dsl.abstraction import AbstractionProduction
from dreamcoder.neurosym.types.type_signature import FunctionTypeSignature
from dreamcoder.neurosym.types.type_with_environment import Environment, TypeWithEnvironment
from dreamcoder.tests.program_dist.utils import enumerate_dsl
from dreamcoder.program import Program
import os
import traceback
import subprocess
import numpy as np
import time as time

DEFAULT_SOLVER_DIRECTORY = "."

INDUCTIVE_EXAMPLES_LIKELIHOOD_MODEL = "inductive_examples_likelihood_model"  # Only use the inductive examples to determine the likelihood.

INDUCTIVE_EXAMPLES_DISCOUNTED_PRIOR_LIKELIHOOD_MODEL = "inductive_examples_discounted_prior_likelihood_model"  # Use the inductive examples or otherwise use a discounted prior

def budgetIncrement(lb):
        if True:
            return 1.5
        # Very heuristic - not sure what to do here
        if lb < 24.0:
            return 1.0
        elif lb < 27.0:
            return 0.5
        else:
            return 0.25

import neurosym as ns
from neurosym.dsl.abstraction import _with_index_parameters

def matchBr(s: str, ind: int) -> int | None:
    """
    Given an opening bracket at position ind in string s, find the  position of the corresponding closing bracket.

    Arguments: 
    s (str): denoting the solution program expression (already processed by replacements())
    ind (int): is an integer denoting the starting position of the start bracket '('

    Returns: 
    int | None: an integer denoting position of closing bracket. If start index does not have an open bracket or no closing brackets close the starting bracket, returns None.
    """
    brPair = 0
    for j in range(ind, len(s)):
        if s[j]=="(":
            brPair+=1
        if s[j]==")":
            brPair-=1
        if brPair==0:
            if j==ind:
                return None
            return j
    return None

def get_argument_list(s: str) -> list[str] | None:
    """
    Returns function body and arguments present in a lambda function serving as an abstraction in DreamCoder's grammar.

    Args:
        s (str): function body

    Returns:
        list[str] | None: lambda function body and arguments present in the function. If there is neither body nor a set of arguments, this returns None.
    """
    start_bracket = s.find("(")
    if start_bracket != -1:
        end_bracket = matchBr(s, start_bracket)
        if end_bracket == None:
            print(f"No closing bracket found for {s}")
            return None
        remaining_arguments = get_argument_list(s[end_bracket+2:])
        if remaining_arguments == None:
            return [s[start_bracket+1:end_bracket]]
        else:
            return [s[start_bracket+1:end_bracket]] + remaining_arguments
    else:
        return s.split(" ")
    
def parse_abstraction_dc_to_ns(abstraction: str, primitive_list: list[str]) -> ns.InitializedSExpression | ns.AbstractionIndexParameter | None:
    """
    Converts a DreamCoder (Stitch) abstraction or a subabstraction within it to a NeuroSym InitializedSExpression or a AbstractionIndexParameter.
    
    Example of such a Stitch abstraction string: 
    "#(lambda (lambda (#(lambda (mathDomain_swap (mathDomain_simplify (mathDomain_rrotate (mathDomain_swap (mathDomain_div (mathDomain_swap (mathDomain_simplify (mathDomain_rrotate $0 mathDomain_4) mathDomain_0) mathDomain_0) mathDomain_3) mathDomain_5) mathDomain_4) mathDomain_0) mathDomain_0)) (mathDomain_swap (#(lambda (mathDomain_simplify (mathDomain_dist (mathDomain_rrotate (mathDomain_sub $0 mathDomain_5) mathDomain_1) mathDomain_1) mathDomain_0)) (mathDomain_multone (mathDomain_rrotate (#(lambda (mathDomain_simplify (mathDomain_dist (mathDomain_rrotate (mathDomain_sub $0 mathDomain_5) mathDomain_1) mathDomain_1) mathDomain_0)) (mathDomain_swap (#(lambda (mathDomain_swap (mathDomain_simplify $0 mathDomain_0) mathDomain_0)) $0) $1)) mathDomain_4) mathDomain_5)) mathDomain_5))))"
    
    Example output of such a conversion:
    (lambda (mathDomain_swap (mathDomain_simplify (mathDomain_rrotate (mathDomain_swap (mathDomain_div (mathDomain_swap (mathDomain_simplify (mathDomain_rrotate (mathDomain_swap (mathDomain_simplify (mathDomain_dist (mathDomain_rrotate (mathDomain_sub (mathDomain_multone (mathDomain_rrotate (mathDomain_simplify (mathDomain_dist (mathDomain_rrotate (mathDomain_sub (mathDomain_swap (mathDomain_swap (mathDomain_simplify #0 (mathDomain_0)) (mathDomain_0)) #1) (mathDomain_5)) (mathDomain_1)) (mathDomain_1)) (mathDomain_0)) (mathDomain_4)) (mathDomain_5)) (mathDomain_5)) (mathDomain_1)) (mathDomain_1)) (mathDomain_0)) (mathDomain_5)) (mathDomain_4)) (mathDomain_0)) (mathDomain_0)) (mathDomain_3)) (mathDomain_5)) (mathDomain_4)) (mathDomain_0)) (mathDomain_0)))
    
    Args:
        abstraction (str): Abstraction discovered by Stitch in DreamCoder
    
    Returns:
        ns.InitializedSExpression |  ns.AbstractionIndexParameter | None: Abstraction in NeuroSym format. If the abstraction is not parseable, returns None.
    """
    if abstraction == "":
        print("Empty abstraction")
        return None
    if abstraction[0] == "(":
        if matchBr(abstraction, 0) == len(abstraction)-1:
            return parse_abstraction_dc_to_ns(abstraction[1:-1], primitive_list)
        else:
            print(f"Unmatched brackets in {abstraction}")
            return None
    elif abstraction[0] == "#":
        end_bracket = matchBr(abstraction, 1)
        if end_bracket == len(abstraction)-1:
            return parse_abstraction_dc_to_ns(abstraction[2:-1], primitive_list)
        else:
            if end_bracket == None:
                print(f"No closing bracket found for {abstraction}")
                return None
            argument_list = get_argument_list(abstraction[1:])
            if argument_list == None:
                print(f"No arguments found for {abstraction}")
                return None
            arg_tuple = tuple([parse_abstraction_dc_to_ns(arg_abstraction, primitive_list) for arg_abstraction in argument_list])
            if None in arg_tuple:
                print(f"None in arg_tuple for {abstraction}")
            arg_tuple = tuple([x for x in arg_tuple if x != None])
            if len(arg_tuple) == 0:
                return None
            if len(arg_tuple) == 1:
                return arg_tuple[0]
            sub_abstraction_body = arg_tuple[0]
            sub_abstraction_args = arg_tuple[1:]
            #This assert ensures that the function at the root of the InitializedSExpression has the correct types
            assert isinstance(sub_abstraction_body, ns.InitializedSExpression)
            result = _with_index_parameters(sub_abstraction_body, sub_abstraction_args, True)
            #This assert ensures that the final result of calling _with_index_parameters has the correct types and not just an "object" type (derived from the apply function call in the _with_index_parameters function)
            assert isinstance(result, ns.InitializedSExpression) or isinstance(result, ns.AbstractionIndexParameter)
            return result
            # return ns.InitializedSExpression("lambda", arg_tuple, {})
    elif abstraction.split(" ")[0] == "lambda":
        end_bracket = matchBr(abstraction, 7)
        if end_bracket  == len(abstraction)-1:
            return parse_abstraction_dc_to_ns(abstraction[8:-1], primitive_list)
        else:
            if end_bracket == None:
                print(f"No closing bracket found for {abstraction}")
                return None
            argument_list = get_argument_list(abstraction[7:end_bracket])
            if argument_list == None:
                print(f"No arguments found for {abstraction}")
                return None
            arg_tuple = tuple([parse_abstraction_dc_to_ns(arg_abstraction, primitive_list) for arg_abstraction in argument_list])
            if None in arg_tuple:
                print(f"None in arg_tuple for {abstraction}")
            arg_tuple = tuple([x for x in arg_tuple if x != None])
            return ns.InitializedSExpression("lambda", arg_tuple, {})
    else:
        splits = abstraction.split(" ")
        func = splits[0]
        if func in primitive_list:
            argument_list = get_argument_list(abstraction[len(func)+1:])
            if argument_list == None:
                print(f"No arguments found for {abstraction}")
                return None
            arg_tuple = tuple([parse_abstraction_dc_to_ns(arg_abstraction, primitive_list) for arg_abstraction in argument_list])
            if None in arg_tuple:
                print(f"None in arg_tuple for {abstraction}")
            arg_tuple = tuple([x for x in arg_tuple if x != None])
            return ns.InitializedSExpression(func, arg_tuple, {})
        else:
            if abstraction[0] == "$":
                abstraction_child = ns.AbstractionIndexParameter(int(abstraction.split(" ")[0][1:]))
                return abstraction_child
            else:
                return ns.InitializedSExpression(abstraction, (), {})

def multicoreEnumeration(
    g,
    tasks,
    _=None,
    enumerationTimeout=None,
    solver="ocaml",
    CPUs=1,
    maximumFrontier=None,
    verbose=True,
    evaluationTimeout=None,
    testing=False,
    unigramGrammar=None,
    max_mem_per_enumeration_thread=1000000,
    solver_directory=DEFAULT_SOLVER_DIRECTORY,
    likelihood_model_string=INDUCTIVE_EXAMPLES_LIKELIHOOD_MODEL,
):
    """g: Either a Grammar, or a map from task to grammar.
    Returns (list-of-frontiers, map-from-task-to-search-time)"""
    
    assert enumerationTimeout is not None, "enumerateForTasks: You must provide a timeout."
    
    if not isinstance(g, dict):
        g = {t: g for t in tasks}
    task2grammar = g
    print(f"task2grammar: {task2grammar}")
    for t in tasks:
        print(f"Expressions2Likelihood: {task2grammar[t].expression2likelihood}")
        break
    dslf = ns.DSLFactory()
    
    #Define neurosym-equivalent DSL here. Assert that it has the same primitive names as the DreamCoder DSL
    dslf.concrete("1", "() -> i", lambda: 1)
    dslf.concrete("incr", "(i) -> i", lambda x: x + 1)
    dslf.concrete("incr2", "(i) -> i", lambda x: x + 2)
    dslf.prune_to("i")
    max_arity = 1
    num_productions = 4 # make sure to include root in this count
    dsl = dslf.finalize()
    primitive_list = [prod[0] for prod in dslf._concrete_productions]
    
    for task in g.keys():
        for k, v in task2grammar[task].expression2likelihood.items():
            if k not in primitive_list:
                #This means that k is likely to be an abstraction, so we generate an abstraction production from the DreamCoder abstraction as described in the grammar
                parsed_abstraction = parse_abstraction_dc_to_ns(k, primitive_list)
                if parsed_abstraction != None:
                    s_exp = ns.SExpression(k, [parsed_abstraction])
                    type_argument = [dsl.compute_type_abs(x) for x in s_exp.children][0]
                    reversed_order_items = [value for _, value in type_argument.env._elements.items()]
                    new_type_env_dict = {}
                    for key in type_argument.env._elements.keys():
                        new_type_env_dict[key] = reversed_order_items.pop()
                    corrected_type_argument = TypeWithEnvironment(typ = type_argument.typ, env = Environment(_elements = new_type_env_dict))
                    type_signature = FunctionTypeSignature([x[0] for _, x in corrected_type_argument.env._elements.items()], corrected_type_argument.typ)
                    final = AbstractionProduction(k, type_signature, s_exp)                    
                    dsl = dsl.add_production(final)                    
                    primitive_list.append(k)
    
    #Now that we have the neurosym-equivalent DSL, we can create the BigramProgramDistributionFamily
    family = ns.BigramProgramDistributionFamily(dsl)
    print(f"Bigram Parameters Shape is: {family.parameters_shape()}")
    frontiers = []
    dist_dict = {task: np.zeros((num_productions, max_arity, num_productions), dtype=np.float32) for task in tasks}
    for t in tasks:
        dist = np.zeros((num_productions, max_arity, num_productions), dtype=np.float32)
        likelihood_dict = task2grammar[t].expression2likelihood
        dreamcoder_ns_mapping = {}
        #Automatically maps dreamcoder primitive name to corresponding neurosym index
        ordered_symbols = dsl.ordered_symbols(include_root=True)
        for p in range(len(ordered_symbols)):
            dreamcoder_ns_mapping[ordered_symbols[p]] = p
        for k, v in likelihood_dict.items():
           production_ind = dreamcoder_ns_mapping[k]
           for prod_ind in range(num_productions):
               for arity in range(max_arity):
                   dist[prod_ind][arity][production_ind] = v
        #Filter the productions out that are impossible to reach
        dist_dict[t] = dist * family._valid_mask
    
    min_likelihood_dict = {task: 0.0 for task in tasks}
    enumerations = {task: [] for task in tasks}
    
    #SAGNIK_TBD: Recreate the while loop below - while satisfying memory and time constraints, enumerate all programs in between the lower and upper bound
    # SAGNIK_TBD: Optimize for one CPU per job
    for job in min_likelihood_dict.keys():
        starting = time.time()
        parsed_enumerations = []
        while True:
            bi = budgetIncrement(min_likelihood_dict[job])
            current_generations = list(family.enumerate(dist = dist_dict[job], min_likelihood = min_likelihood_dict[job], chunk_size = bi))
            enumerations[job] += current_generations
            
            #Here we attempt to parse enumerated programs to DreamCoder-style programs to create DreamCoder Frontier Entries
            for ns_prog_tuple in current_generations:
                ns_prog, prob_fraction = ns_prog_tuple
                dreamcoder_prog = Program.parse(ns_prog)
                log_prob = math.log(float(format(prob_fraction, '.10g')))
                dreamcoder_entry = FrontierEntry(program=dreamcoder_prog, logLikelihood=log_prob)
                parsed_enumerations.append(dreamcoder_entry)
            min_likelihood_dict[job] += bi
            if time.time() - starting > enumerationTimeout:
                break
        frontiers[job] = Frontier(parsed_enumerations, task=t)        
    
    bestSearchTime = {t: None for t in task2grammar}
    return [frontiers[t] for t in tasks], bestSearchTime

    #use expression2likelihood for bigram in dsl
    enumerations = list(enumerate_dsl(family, dist, min_likelihood=-1000000))
    print(enumerations)
    # for t in tasks:
    #     frontier = Frontier(
    #         [
    #             FrontierEntry(
    #                 program = Program.parse(e[0]),
    #                 logLikelihood= e[1].as_integer_ratio()[0]/e[1].as_integer_ratio()[1],
    #                 #tokens= escape_tokens(e["tokens"]).split(),
    #                 logPrior= g.logLikelihood(t.request, Program.parse(e[0])), 
    #             )
    #             for e in enumerations
    #         ],
    #         task=t,
    #     )

    #frontiers = {t: Frontier([Program.parse(i[0]) for i in enumerations], task=t) for t in task2grammar}
    frontiers = {t: Frontier([], task=t) for t in task2grammar}
    bestSearchTime = {t: None for t in task2grammar}

    # # We don't use actual threads but instead use the multiprocessing
    # # library. This is because we need to be able to kill workers.
    # # from multiprocess import Process, Queue
    # print(
    #     f"Beginning enumeration on a total of {len(tasks)} tasks with a total of {CPUs} CPUs."
    # )
    # from multiprocessing import Queue

    # # everything that gets sent between processes will be dilled
    # import dill

    # solvers = {
    #     "ocaml": solveForTask_ocaml,
    #     "pypy": solveForTask_pypy,
    #     "python": solveForTask_python,
    # }
    # assert (
    #     solver in solvers
    # ), "You must specify a valid solver. options are ocaml, pypy, or python."

    # likelihoodModel = None
    # if solver == "pypy" or solver == "python":
    #     # Use an all or nothing likelihood model.
    #     likelihoodModel = AllOrNothingLikelihoodModel(timeout=evaluationTimeout)
    # if solver == "ocaml":
    #     likelihoodModel = likelihood_model_string

    # solver = solvers[solver]

    # if not isinstance(g, dict):
    #     g = {t: g for t in tasks}
    # task2grammar = g

    # # If we are not evaluating on held out testing tasks:
    # # Bin the tasks by request type and grammar
    # # If these are the same then we can enumerate for multiple tasks simultaneously
    # # If we are evaluating testing tasks:
    # # Make sure that each job corresponds to exactly one task
    # jobs = {}
    # for i, t in enumerate(tasks):
    #     if testing:
    #         k = (task2grammar[t], t.request, i)
    #     else:
    #         k = (task2grammar[t], t.request)
    #     jobs[k] = jobs.get(k, []) + [t]

    # disableParallelism = len(jobs) == 1
    # parallelCallback = (
    #     launchParallelProcess
    #     if not disableParallelism
    #     else lambda f, *a, **k: f(*a, **k)
    # )
    # if disableParallelism:
    #     eprint("Disabling parallelism on the Python side because we only have one job.")
    #     eprint("If you are using ocaml, there could still be parallelism.")

    # # Map from task to the shortest time to find a program solving it
    # bestSearchTime = {t: None for t in task2grammar}

    # lowerBounds = {k: 0.0 for k in jobs}

    # frontiers = {t: Frontier([], task=t) for t in task2grammar}

    # # For each job we keep track of how long we have been working on it
    # stopwatches = {t: Stopwatch() for t in jobs}

    # # Map from task to how many programs we enumerated for that task
    # taskToNumberOfPrograms = {t: 0 for t in tasks}

    # def numberOfHits(f):
    #     return sum(e.logLikelihood > -0.01 for e in f)

    # def budgetIncrement(lb):
    #     if True:
    #         return 1.5
    #     # Very heuristic - not sure what to do here
    #     if lb < 24.0:
    #         return 1.0
    #     elif lb < 27.0:
    #         return 0.5
    #     else:
    #         return 0.25

    # def maximumFrontiers(j):
    #     tasks = jobs[j]
    #     return {t: maximumFrontier - numberOfHits(frontiers[t]) for t in tasks}

    # def allocateCPUs(n, tasks):
    #     allocation = {t: 0 for t in tasks}
    #     while n > 0:
    #         for t in tasks:
    #             # During testing we use exactly one CPU per task
    #             if testing and allocation[t] > 0:
    #                 return allocation
    #             allocation[t] += 1
    #             n -= 1
    #             if n == 0:
    #                 break
    #     return allocation

    # def refreshJobs():
    #     for k in list(jobs.keys()):
    #         v = [
    #             t
    #             for t in jobs[k]
    #             if numberOfHits(frontiers[t]) < maximumFrontier
    #             and stopwatches[k].elapsed <= enumerationTimeout
    #         ]
    #         if v:
    #             jobs[k] = v
    #         else:
    #             del jobs[k]

    # # Workers put their messages in here
    # q = Queue()

    # # How many CPUs are we using?
    # activeCPUs = 0

    # # How many CPUs was each job allocated?
    # id2CPUs = {}
    # # What job was each ID working on?
    # id2job = {}
    # nextID = 0

    # while True:
    #     refreshJobs()
    #     # Don't launch a job that we are already working on
    #     # We run the stopwatch whenever the job is being worked on
    #     # freeJobs are things that we are not working on but could be
    #     freeJobs = [
    #         j
    #         for j in jobs
    #         if not stopwatches[j].running
    #         and stopwatches[j].elapsed < enumerationTimeout - 0.5
    #     ]
    #     if freeJobs and activeCPUs < CPUs:
    #         # Allocate a CPU to each of the jobs that we have made the least
    #         # progress on
    #         freeJobs.sort(key=lambda j: lowerBounds[j])
    #         # Launch some more jobs until all of the CPUs are being used
    #         availableCPUs = CPUs - activeCPUs
    #         allocation = allocateCPUs(availableCPUs, freeJobs)
    #         for j in freeJobs:
    #             if allocation[j] == 0:
    #                 continue
    #             g, request = j[:2]
    #             bi = budgetIncrement(lowerBounds[j])
    #             thisTimeout = enumerationTimeout - stopwatches[j].elapsed
    #             eprint(
    #                 "(python) Launching %s (%d tasks) w/ %d CPUs. %f <= MDL < %f. Timeout %f."
    #                 % (
    #                     request,
    #                     len(jobs[j]),
    #                     allocation[j],
    #                     lowerBounds[j],
    #                     lowerBounds[j] + bi,
    #                     thisTimeout,
    #                 )
    #             )
    #             stopwatches[j].start()
    #             parallelCallback(
    #                 wrapInThread(solver),
    #                 q=q,
    #                 g=g,
    #                 ID=nextID,
    #                 elapsedTime=stopwatches[j].elapsed,
    #                 CPUs=allocation[j],
    #                 tasks=jobs[j],
    #                 lowerBound=lowerBounds[j],
    #                 upperBound=lowerBounds[j] + bi,
    #                 budgetIncrement=bi,
    #                 timeout=thisTimeout,
    #                 evaluationTimeout=evaluationTimeout,
    #                 maximumFrontiers=maximumFrontiers(j),
    #                 testing=testing,
    #                 likelihoodModel=likelihoodModel,
    #                 unigramGrammar=unigramGrammar,
    #                 max_mem_per_enumeration_thread=max_mem_per_enumeration_thread,
    #                 solver_directory=solver_directory,
    #             )
    #             id2CPUs[nextID] = allocation[j]
    #             id2job[nextID] = j
    #             nextID += 1

    #             activeCPUs += allocation[j]
    #             lowerBounds[j] += bi

    #     # If nothing is running, and we just tried to launch jobs,
    #     # then that means we are finished
    #     if all(not s.running for s in stopwatches.values()):
    #         break

    #     # Wait to get a response
    #     message = Bunch(dill.loads(q.get()))

    #     if message.result == "failure":
    #         eprint("PANIC! Exception in child worker:", message.exception)
    #         eprint(message.stacktrace)
    #         assert False
    #     elif message.result == "success":
    #         # Mark the CPUs is no longer being used and pause the stopwatch
    #         activeCPUs -= id2CPUs[message.ID]
    #         stopwatches[id2job[message.ID]].stop()

    #         newFrontiers, searchTimes, pc = message.value
    #         for t, f in newFrontiers.items():
    #             oldBest = None if len(frontiers[t]) == 0 else frontiers[t].bestPosterior
    #             frontiers[t] = frontiers[t].combine(f)
    #             newBest = None if len(frontiers[t]) == 0 else frontiers[t].bestPosterior

    #             taskToNumberOfPrograms[t] += pc

    #             dt = searchTimes[t]
    #             if dt is not None:
    #                 if bestSearchTime[t] is None:
    #                     bestSearchTime[t] = dt
    #                 else:
    #                     # newBest & oldBest should both be defined
    #                     assert oldBest is not None
    #                     assert newBest is not None
    #                     newScore = newBest.logPrior + newBest.logLikelihood
    #                     oldScore = oldBest.logPrior + oldBest.logLikelihood

    #                     if newScore > oldScore:
    #                         bestSearchTime[t] = dt
    #                     elif newScore == oldScore:
    #                         bestSearchTime[t] = min(bestSearchTime[t], dt)
    #     else:
    #         eprint("Unknown message result:", message.result)
    #         assert False

    # eprint(
    #     "We enumerated this many programs, for each task:\n\t",
    #     list(taskToNumberOfPrograms.values()),
    # )

    # return [frontiers[t] for t in tasks], bestSearchTime


def wrapInThread(f):
    """
    Returns a function that is designed to be run in a thread/threadlike process.
    Result will be either put into the q
    """
    import dill

    def _f(*a, **k):
        q = k.pop("q")
        ID = k.pop("ID")

        try:
            r = f(*a, **k)
            q.put(dill.dumps({"result": "success", "ID": ID, "value": r}))
        except Exception as e:
            q.put(
                dill.dumps(
                    {
                        "result": "failure",
                        "exception": e,
                        "stacktrace": traceback.format_exc(),
                        "ID": ID,
                    }
                )
            )
            return

    return _f


OCAML_TEST_FLAG = "is_ocaml_test"  # Indicates a JSON response intended for testing.


def solveForTask_ocaml(
    _=None,
    elapsedTime=0.0,
    CPUs=1,
    g=None,
    tasks=None,
    lowerBound=None,
    upperBound=None,
    budgetIncrement=None,
    timeout=None,
    testing=None,  # FIXME: unused
    likelihoodModel=None,
    evaluationTimeout=None,
    maximumFrontiers=None,
    unigramGrammar=None,
    verbose=False,
    max_mem_per_enumeration_thread=1000000,
    solver_directory=DEFAULT_SOLVER_DIRECTORY,
):

    import json

    def taskMessage(t):
        serialized_examples = []
        for xs, y in t.examples:
            if hasattr(t, "serializeSpecialInput"):
                xs = t.serializeSpecialInput(xs)
            if hasattr(t, "serializeSpecialOutput"):
                y = t.serializeSpecialOutput(y, is_output=True)
            serialized_examples.append({"inputs": list(xs), "output": y})

        m = {
            "examples": serialized_examples,
            "name": t.name,
            "request": t.request.json(),
            "maximumFrontier": maximumFrontiers[t],
        }
        if hasattr(t, "specialTask"):
            special, extra = t.specialTask
            m["specialTask"] = special
            m["extras"] = extra
        if hasattr(t, "raw_programs_to_test"):
            m["raw_programs_to_test"] = t.raw_programs_to_test
        return m

    message = {
        "DSL": g.json(),
        "tasks": [taskMessage(t) for t in tasks],
        "programTimeout": float(evaluationTimeout),
        "nc": CPUs,
        "timeout": timeout,
        "lowerBound": lowerBound,
        "upperBound": upperBound,
        "budgetIncrement": budgetIncrement,
        "verbose": verbose,
        "shatter": 5 if len(tasks) == 1 and "turtle" in str(tasks[0].request) else 10,
        "likelihoodModel": likelihoodModel,
    }

    if hasattr(tasks[0], "maxParameters") and tasks[0].maxParameters is not None:
        message["maxParameters"] = tasks[0].maxParameters

    message = json.dumps(message)
    # uncomment this if you want to save the messages being sent to the solver

    solver_file = "solver"
    if hasattr(tasks[0], "specialSolver"):
        solver_file = tasks[0].specialSolver

    try:
        solver_file = os.path.join(get_root_dir(), solver_directory, solver_file)
        process = subprocess.Popen(
            solver_file, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        limit_virtual_memory_with_psutil_if_possible(
            process, max_mem_per_enumeration_thread
        )

        response, error = process.communicate(bytes(message, encoding="utf-8"))
        response = json.loads(response.decode("utf-8"))
    except OSError as exc:
        raise exc

    except:
        print("response:", response)
        print("error:", error)
        with open("message", "w") as f:
            f.write(message)
        # Don't fail on errors
        # assert False, "MAX RAISE"
        print(
            "ERROR in enumeration, returning empty frontiers for this batch of tasks."
        )
        response = {t.name: [] for t in tasks}  # Empty response

    def escape_tokens(tokens):
        if unigramGrammar is not None:
            return unigramGrammar.escape_tokens_string(tokens)
        return g.escape_tokens_string(tokens)

    if OCAML_TEST_FLAG in response:
        return response

    pc = response.get("number_enumerated", 0)  # TODO
    frontiers = {}
    searchTimes = {}
    for t in tasks:
        solutions = response[t.name]
        frontier = Frontier(
            [
                FrontierEntry(
                    program=p,
                    logLikelihood=e["logLikelihood"],
                    tokens=escape_tokens(e["tokens"]).split(),
                    logPrior=g.logLikelihood(t.request, p),
                )
                for e in solutions
                for p in [Program.parse(e["program"])]
            ],
            task=t,
        )
        frontiers[t] = frontier
        if frontier.empty:
            searchTimes[t] = None
        # This is subtle:
        # The search time we report is actually not be minimum time to find any solution
        # Rather it is the time to find the MAP solution
        # This is important for regression problems,
        # where we might find something with a good prior but bad likelihood early on,
        # and only later discovered the good high likelihood program
        else:
            searchTimes[t] = (
                min((e["logLikelihood"] + e["logPrior"], e["time"]) for e in solutions)[
                    1
                ]
                + elapsedTime
            )

    return frontiers, searchTimes, pc


def solveForTask_pypy(
    _=None,
    elapsedTime=0.0,
    g=None,
    task=None,
    lowerBound=None,
    upperBound=None,
    budgetIncrement=None,
    timeout=None,
    likelihoodModel=None,
    evaluationTimeout=None,
    maximumFrontier=None,
    testing=False,
    unigramGrammar=None,
):
    return callCompiled(
        enumerateForTasks,
        g,
        tasks,
        likelihoodModel,
        timeout=timeout,
        testing=testing,
        elapsedTime=elapsedTime,
        evaluationTimeout=evaluationTimeout,
        maximumFrontiers=maximumFrontiers,
        budgetIncrement=budgetIncrement,
        lowerBound=lowerBound,
        upperBound=upperBound,
        unigramGrammar=None,
    )


def solveForTask_python(
    _=None,
    elapsedTime=0.0,
    g=None,
    tasks=None,
    lowerBound=None,
    upperBound=None,
    budgetIncrement=None,
    timeout=None,
    CPUs=1,
    likelihoodModel=None,
    evaluationTimeout=None,
    maximumFrontiers=None,
    testing=False,
    unigramGrammar=None,
):
    return enumerateForTasks(
        g,
        tasks,
        likelihoodModel,
        timeout=timeout,
        testing=testing,
        elapsedTime=elapsedTime,
        evaluationTimeout=evaluationTimeout,
        maximumFrontiers=maximumFrontiers,
        budgetIncrement=budgetIncrement,
        lowerBound=lowerBound,
        upperBound=upperBound,
        unigramGrammar=None,
    )


class EnumerationTimeout(Exception):
    pass


def enumerateForTasks(
    g,
    tasks,
    likelihoodModel,
    _=None,
    verbose=False,
    timeout=None,
    elapsedTime=0.0,
    CPUs=1,
    testing=False,  # unused
    evaluationTimeout=None,
    lowerBound=0.0,
    upperBound=100.0,
    budgetIncrement=1.0,
    maximumFrontiers=None,
    unigramGrammar=None,
):
    assert timeout is not None, "enumerateForTasks: You must provide a timeout."

    from time import time

    request = tasks[0].request
    assert all(
        t.request == request for t in tasks
    ), "enumerateForTasks: Expected tasks to all have the same type"

    maximumFrontiers = [maximumFrontiers[t] for t in tasks]
    # store all of the hits in a priority queue
    # we will never maintain maximumFrontier best solutions
    hits = [PQ() for _ in tasks]

    starting = time()
    previousBudget = lowerBound
    budget = lowerBound + budgetIncrement
    try:
        totalNumberOfPrograms = 0
        while (
            time() < starting + timeout
            and any(len(h) < mf for h, mf in zip(hits, maximumFrontiers))
            and budget <= upperBound
        ):
            numberOfPrograms = 0

            for prior, _, p in g.enumeration(
                Context.EMPTY,
                [],
                request,
                maximumDepth=99,
                upperBound=budget,
                lowerBound=previousBudget,
            ):
                descriptionLength = -prior
                # Shouldn't see it on this iteration
                assert descriptionLength <= budget
                # Should already have seen it
                assert descriptionLength > previousBudget

                numberOfPrograms += 1
                totalNumberOfPrograms += 1

                for n in range(len(tasks)):
                    task = tasks[n]

                    # Warning:changed to max's new likelihood model situation
                    # likelihood = task.logLikelihood(p, evaluationTimeout)
                    # if invalid(likelihood):
                    # continue
                    success, likelihood = likelihoodModel.score(p, task)
                    if not success:
                        continue

                    dt = time() - starting + elapsedTime
                    priority = -(likelihood + prior)
                    hits[n].push(
                        priority,
                        (
                            dt,
                            FrontierEntry(
                                program=p, logLikelihood=likelihood, logPrior=prior
                            ),
                        ),
                    )
                    if len(hits[n]) > maximumFrontiers[n]:
                        hits[n].popMaximum()

                if timeout is not None and time() - starting > timeout:
                    raise EnumerationTimeout

            previousBudget = budget
            budget += budgetIncrement

            if budget > upperBound:
                break
    except EnumerationTimeout:
        pass
    frontiers = {
        tasks[n]: Frontier([e for _, e in hits[n]], task=tasks[n])
        for n in range(len(tasks))
    }
    searchTimes = {
        tasks[n]: None if len(hits[n]) == 0 else min(t for t, _ in hits[n])
        for n in range(len(tasks))
    }

    return frontiers, searchTimes, totalNumberOfPrograms
