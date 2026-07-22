from dreamcoder.program import Primitive
from dreamcoder.type import arrow, tint, tstr
import math

ops = ["+", "-", "*", "/"]

class Tree:

  def __init__(self, root, left="None", right="None") -> None:
    #Initialization of tree with root value and left and right subtrees
    self.root = root
    self.left = left
    self.right = right

  def __eq__(self, __o: object) -> bool:
    #Enables equality comparison of trees
    return isinstance(
      __o, Tree
    ) and self.root == __o.root and self.left == __o.left and self.right == __o.right

  def __repr__(self) -> str:
    #String representation of the tree
    return "(" + str(self.root) + " " + str(self.left) + " " + str(
      self.right) + ")"

  def __typerepr__(self) -> str:
    #Shows the types present in the tree, mostly concerned with str/int working, hence the [-6:-1] index range
    leftType = str(type(self.left))
    rightType = str(type(self.right))
    if (leftType == "<class '__main__.Tree'>"):
      leftType = self.left.__typerepr__()
    if (rightType == "<class '__main__.Tree'>"):
      rightType = self.right.__typerepr__()
    return "(" + str(type(
      self.root))[-6:-1] + " " + leftType + " " + rightType + ")"


def isNum(x):
  return (isinstance(x, int) or isinstance(x, float))


def intConvertable(s):
  try:
    int(s)
    return True
  except ValueError:
    return False


def floatConvertable(s):
  try:
    float(s)
    return True
  except ValueError:
    return False


def detreefy(tree):
  #converts tree into prefix equation string
  if tree == "None":
    return ""
  left = detreefy(tree.left)
  right = detreefy(tree.right)
  if left != "":
    left = " " + str(left)
  if right != "":
    right = " " + str(right)
  return "(" + str(tree.root) + str(left) + str(right) + ")"


def matchBracket(string, ind):
  #finds corresponding matching bracket of the bracket in prefix notation
  if string[ind] == "(":
    brCount = 1
    for i in range(ind + 1, len(string)):
      if (string[i] == "("):
        brCount += 1
      if (string[i] == ")"):
        brCount -= 1
      if brCount == 0:
        return i
  elif string[ind] == ")":
    brCount = 1
    for i in range(ind - 1, 0, -1):
      if (string[i] == "("):
        brCount -= 1
      if (string[i] == ")"):
        brCount += 1
      if brCount == 0:
        return i
  return "Error"


def treefy(eq):
  #converts prefix equation string to tree
  #operations must be in ops or "="
  #numbers must be int or floats
  #no space between brackets and operations succeeding them
  #(+ (- (x) (3)) (y))
  if eq == "None":
    return "None"
  newEq = eq[1:-1]
  fstArgInd = newEq.find("(")
  sndArgInd = newEq.rfind(")")
  fstArgMatch = "None"
  sndArgMatch = "None"
  args = [newEq.split(" ")[0]]
  if fstArgInd != -1:
    fstArgMatch = matchBracket(newEq, fstArgInd)
  if sndArgInd != -1:
    sndArgMatch = matchBracket(newEq, sndArgInd)
  if (fstArgInd != -1 and sndArgInd != -1):
    if (fstArgInd != sndArgMatch):
      try:
        args.append(newEq[fstArgInd:fstArgMatch + 1])
        args.append(newEq[sndArgMatch:sndArgInd + 1])
      except Exception as e:
        print(e)
        print(f"treefy failed on {newEq}")
    else:
      args.append(newEq[fstArgInd:fstArgMatch + 1])
  if (intConvertable(args[0])):
    args[0] = int(args[0])
  elif (floatConvertable(args[0])):
    args[0] = float(args[0])
  while len(args) < 3:
    args.append("None")
  return Tree(args[0], treefy(args[1]), treefy(args[2]))

def permit_rotations(operation_tuple):
    """
    Checks if operations are eligible for rotation. E.g. (x+5)-5 should resolve to (x+(5-5)), but 2-5+3 cannot resolve to 2-8. (5-x)+5 cannot resolve to 5-(x+5). 

    Input: A tuple of two operations, op1 and op2, where op1 is P in the diagram at https://en.wikipedia.org/wiki/File:Tree_rotation.png, while op2 is Q, passed in the format (op1, op2). 
    Output: A tuple of three values: (T/F, new_P_1, new_Q_2)
    """
    permissible_rotations = {
    ("-", "-"): (True, "-", "+"),
    ("-", "+"): (True, "-", "-"),
    ("+", "+"): (True, "+", "+"),
    ("+", "-"): (True, "+", "-"),
    ("*", "*"): (True, "*", "*"),
    ("*", "/"): (True, "*", "/"),
    ("/", "/"): (True, "/", "*"),
    ("/", "*"): (True, "/", "/")
    }
    if operation_tuple in permissible_rotations.keys():
      return permissible_rotations[operation_tuple]
    else:
      return (False, operation_tuple[0], operation_tuple[1])

def _rrotateHelper(s):
  #Right rotation of tree
  #originalRoot = Q
  #upperLeft = P
  #upperRight = C
  #bottomLeft = A
  #bottomRight = B
  #follow diagram at https://en.wikipedia.org/wiki/File:Tree_rotation.png
  eqTree = treefy(s)
  if eqTree.root in ops and eqTree.left.root in ops:
    newOps = permit_rotations((eqTree.left.root, eqTree.root))
    if newOps[0]:
        upperLeft = newOps[1]
        originalRoot = newOps[2]
        upperRight = eqTree.right
        bottomLeft = eqTree.left.left
        bottomRight = eqTree.left.right
        rightSub = Tree(originalRoot, bottomRight, upperRight)
        newTree = Tree(upperLeft, bottomLeft, rightSub)
        return detreefy(newTree)
  return s

def _lrotateHelper(s):
  #Left rotation of tree
  #originalRoot = P
  #upperLeft = A
  #upperRight = Q
  #bottomLeft = B 
  #bottomRight = C
  #follow diagram at https://en.wikipedia.org/wiki/File:Tree_rotation.png
  eqTree = treefy(s)
  if eqTree.root in ops and eqTree.right.root in ops:
    newOps = permit_rotations((eqTree.root, eqTree.right.root))
    if newOps[0]:
        originalRoot = newOps[1]
        upperLeft = eqTree.left
        upperRight = newOps[2]
        bottomLeft = eqTree.right.left
        bottomRight = eqTree.right.right
        leftSub = Tree(originalRoot, upperLeft, bottomLeft)
        newTree = Tree(upperRight, leftSub, bottomRight)
        return detreefy(newTree)
  return s


def _genSub(s):
  #Generates all subtrees of a given tree
  eqTree = treefy(s)
  if s == "None":
    return []
  elif (eqTree.left == "None" and eqTree.right == "None"):
    return [s]
  else:
    left = [] if eqTree.left == "None" else _genSub(detreefy(eqTree.left))
    right = [] if eqTree.right == "None" else _genSub(detreefy(eqTree.right))
    return [s] + left + right

def _size(s):
    #Computes the number of subtrees that are formed by the current subtree
    return len(_genSub(s))

def _metric(s1, s2):
    #Metric to compute the difference in the subtrees formed in step 1 (s1) and step 2 (s2)
    tree1 = treefy(s1)
    tree2 = treefy(s2)
    tree1l = detreefy(tree1.left)
    tree1r = detreefy(tree1.right)
    tree2l = detreefy(tree2.left)
    tree2r = detreefy(tree2.right)
    return max(abs(_size(tree1l) - _size(tree2l)), abs(_size(tree1r) - _size(tree2r)), 1)

def _reconstruct(i, old, newT):
  #Reconstructs a new tree by swapping in newT in the i-th indexed subtree of old. So if subtree "k" is at the i-th index of result of genSub(old), "k" in old gets replaced by new.
  subList = _genSub(old)
  oldT = treefy(old)
  if (i > len(subList)):
    return oldT
  elif (i == 0):
    return treefy(newT)
  else:
    leftLength = len(_genSub(detreefy(oldT.left)))
    if (i <= leftLength):
      return Tree(oldT.root, _reconstruct(i - 1, detreefy(oldT.left), newT),
                  oldT.right)
    else:
      return Tree(oldT.root, oldT.left,
                  _reconstruct(i - 1 - leftLength, detreefy(oldT.right), newT))


def _treeOp(s, i, op):
  #Performs tree operation on i-th subtree
  allSubs = _genSub(s)
  modifiedSub = op(allSubs[i])
  return detreefy(_reconstruct(i, s, modifiedSub))


def _op(s, x, op):
  #Performs operation on x-th indexed subtree on both sides of tree.
  eqTree = treefy(s)
  subTree = _genSub(s)[x]
  if eqTree.root == '=':
    newLeft = Tree(op, eqTree.left, treefy(subTree))
    newRight = Tree(op, eqTree.right, treefy(subTree))
    newTree = Tree("=", newLeft, newRight)
    return detreefy(newTree)
  return s


def _swapHelper(s):
  #Swaps left and right subtrees in a tree
  eqTree = treefy(s)
  newTree = eqTree
  if eqTree.root=="+" or eqTree.root=="*" or eqTree.root=="=":
    newTree = Tree(eqTree.root, eqTree.right, eqTree.left)
  elif eqTree.root=="-":
    newTree = Tree("+", Tree("-", Tree(0), eqTree.right), eqTree.left)
  elif eqTree.root=="/":
    newTree = Tree("*", Tree("/", Tree(1), eqTree.right), eqTree.left)
  return detreefy(newTree)


def _evalTree(op, left, right):
  if op == "+":
    return detreefy(Tree(left + right))
  if op == "-":
    return detreefy(Tree(left - right))
  if op == "*":
    return detreefy(Tree(left * right))
  if op == "/":
    gcd = math.gcd(left, right)
    if right < 0:
      left = -1 * left
      right = -1 * right
    if left%right==0:
      return detreefy(Tree(left // right))
    elif gcd not in [0, 1]:
      return detreefy(Tree("/", Tree(left // gcd), Tree(right//gcd)))
    elif left<0 and right<0:
      return detreefy(Tree("/", Tree(abs(left)), Tree(abs(right))))
  return detreefy(Tree(op, Tree(left), Tree(right)))


def _simplifyHelper(s):
  #Simplifies the tree where possible
  eqTree = treefy(s)
  if (eqTree.left == "None" and eqTree.right == "None"):
    return detreefy(eqTree)
  else:
    left = treefy(_simplifyHelper(detreefy(eqTree.left)))
    right = treefy(_simplifyHelper(detreefy(eqTree.right)))
    leftSimple = left.root
    rightSimple = right.root
    if (isNum(leftSimple) and isNum(rightSimple)):
      return _evalTree(eqTree.root, leftSimple, rightSimple)
    elif (eqTree.root == "+" and leftSimple == 0) or (eqTree.root == "*"
                                                      and leftSimple == 1):
      return detreefy(right)
    elif ((eqTree.root == "+" or eqTree.root == "-") and rightSimple == 0) or (
      (eqTree.root == "*" or eqTree.root == "/") and rightSimple == 1):
      return detreefy(left)
    elif ((eqTree.root == "-" and left == right and left.root != "None")
          or (eqTree.root == "*" and (leftSimple == 0 or rightSimple == 0))):
      return detreefy(Tree(0))
    elif (eqTree.root == "/" and left == right and left.root != "None"):
      return detreefy(Tree(1))
    else:
      return detreefy(Tree(eqTree.root, left, right))

def _distHelper(s):
    eqTree = treefy(s)
    if (eqTree.root == "-" or eqTree.root == "+"):
        if (eqTree.left.root == "*" and eqTree.right.root == "*") or (eqTree.left.root == "/" and eqTree.right.root == "/"):
            if (detreefy(eqTree.left.right) == detreefy(eqTree.right.right)):
                return detreefy(Tree(eqTree.left.root,
                    Tree(eqTree.root, eqTree.left.left, eqTree.right.left),
                    eqTree.left.right))
        if (eqTree.left.root == "*" and eqTree.right.root == "*"):
            if (detreefy(eqTree.left.left) == detreefy(eqTree.right.left)):
                return detreefy(Tree(eqTree.left.root,
                    Tree(eqTree.root, eqTree.left.right, eqTree.right.right),
                    eqTree.left.left))
            if (detreefy(eqTree.left.right) == detreefy(eqTree.right.left)):
                return detreefy(Tree(eqTree.left.root,
                    Tree(eqTree.root, eqTree.left.left, eqTree.right.right),
                    eqTree.left.right))
            if (detreefy(eqTree.left.left) == detreefy(eqTree.right.right)):
                return detreefy(Tree(eqTree.left.root,
                    Tree(eqTree.root, eqTree.left.right, eqTree.right.left),
                    eqTree.left.left))        
    return s

def _revDistHelper(s):
  eqTree = treefy(s)
  if (eqTree.root == "*" or eqTree.root=="/"):
    if (eqTree.left.root=="+" or eqTree.left.root=="-"):
        return detreefy(Tree(eqTree.left.root, Tree(eqTree.root, eqTree.left.left, eqTree.right), Tree(eqTree.root, eqTree.left.right, eqTree.right)))
  if (eqTree.root == "*"):
    if (eqTree.right.root=="+" or eqTree.right.root=="-"):
        return detreefy(Tree(eqTree.right.root, Tree(eqTree.root, eqTree.left, eqTree.right.left), Tree(eqTree.root, eqTree.left, eqTree.right.right)))
  return s

def _newConstGen(a, b, c):
  return ((a*b)+c)

def _divoneHelper(s):
  eqTree = treefy(s)
  return detreefy(Tree("/", eqTree, Tree(1)))

def _multoneHelper(s):
  eqTree = treefy(s)
  return detreefy(Tree("*", eqTree, Tree(1)))

def _addzeroHelper(s):
  eqTree = treefy(s)
  return detreefy(Tree("+", eqTree, Tree(0)))

def _subzeroHelper(s):
  eqTree = treefy(s)
  return detreefy(Tree("-", eqTree, Tree(0)))

def _add(s, x):
  return _op(s, x, "+")

def _sub(s, x):
  return _op(s, x, "-")

def _mult(s, x):
  return _op(s, x, "*")

def _div(s, x):
  return _op(s, x, "/")

def _rrotate(s, i):
  return _treeOp(s, i, _rrotateHelper)

def _lrotate(s, i):
  return _treeOp(s, i, _lrotateHelper)

def _simplify(s, i):
  return _treeOp(s, i, _simplifyHelper)

def _swap(s, i):
  return _treeOp(s, i, _swapHelper)

def _dist(s, i):
  return _treeOp(s, i, _distHelper)

def _revdist(s, i):
  return _treeOp(s, i, _revDistHelper)

def _addzero(s, i):
  return _treeOp(s, i, _addzeroHelper)

def _subzero(s, i):
  return _treeOp(s, i, _subzeroHelper)

def _multone(s, i):
  return _treeOp(s, i, _multoneHelper)

def _divone(s, i):
  return _treeOp(s, i, _divoneHelper)

def mathPrimitives(LARGEST_CONSTANT:int = 10):
    '''
    The largest constant determines the largest constant which is encoded as a primitive in the domain.
    '''
    return [
        Primitive("mathDomain_add", arrow(tstr, tint, tstr), _add),
        Primitive("mathDomain_sub", arrow(tstr, tint, tstr), _sub),
        Primitive("mathDomain_mult", arrow(tstr, tint, tstr), _mult),
        Primitive("mathDomain_div", arrow(tstr, tint, tstr), _div),
        Primitive("mathDomain_rrotate", arrow(tstr, tint, tstr), _rrotate),
        Primitive("mathDomain_lrotate", arrow(tstr, tint, tstr), _lrotate), 
        Primitive("mathDomain_simplify", arrow(tstr, tint, tstr), _simplify),
        Primitive("mathDomain_dist", arrow(tstr, tint, tstr), _dist),
        Primitive("mathDomain_revdist", arrow(tstr, tint, tstr), _revdist),
        Primitive("mathDomain_swap", arrow(tstr, tint, tstr), _swap),
        Primitive("mathDomain_addzero", arrow(tstr, tint, tstr), _addzero),
        Primitive("mathDomain_subzero", arrow(tstr, tint, tstr), _subzero),
        Primitive("mathDomain_multone", arrow(tstr, tint, tstr), _multone),
        Primitive("mathDomain_divone", arrow(tstr, tint, tstr), _divone),
        Primitive("mathDomain_newConstGen", arrow(tint, tint, tint, tint), _newConstGen)
    ] + [Primitive("mathDomain_"+str(x), tint, x) for x in range(0, LARGEST_CONSTANT+1)]
