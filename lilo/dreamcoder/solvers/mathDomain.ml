open Printf
open Str
    
let ops = ["+"; "-"; "*"; "/"]
          
type 'a tree =
  (* Tree structure that stores equations in prefix tree form *)
  | Leaf
  | Node of 'a node 
and 'a node = {
  value: 'a;
  left: 'a tree;
  right: 'a tree
}

let getValue = function
  (* Returns value of a Tree since Tree.value raises error*)
  | Node x -> x.value
  | Leaf -> "Leaf"

let getLeft = function
  (* Returns left subtree of a Tree since Tree.left raises error*)
  | Node x -> x.left
  | Leaf -> Leaf

let getRight = function
  (* Returns right subtree of a Tree since Tree.right raises error*)
  | Node x -> x.right
  | Leaf -> Leaf

let rec getLength = function
  (* Returns number of nodes in tree (including root) *)
  | Leaf -> 0
  | Node x -> 1 + getLength x.left + getLength x.right 

let rec eq = function
  (* Compares two trees and checks if they are equal *)
  | Node x, Node y -> x.value=y.value && eq (x.left, y.left) && eq (x.right, y.right)
  | Leaf, Leaf-> true 
  | _ -> false;;

let rec dtrfy = function
  (* Converts tree to a string representation of form "(op left right)" *)
  | Leaf -> "" 
  | Node x -> "(" ^ x.value ^ (fun x -> if x<>"" then " "^x else x) (dtrfy x.left) ^ (fun x -> if x<>"" then " "^x else x) (dtrfy x.right) ^ ")"

let string_to_list s =
  (* Converts a string to a list of chars *)
  let rec exp i l =
    if i < 0 then l else exp (i - 1) (s.[i] :: l) in
  exp (String.length s - 1) []

let char_array_to_string s = 
  (* Converts a char array to a string *)
  let chars = Array.to_list s in 
    let buf = Buffer.create 16 in
      List.iter (Buffer.add_char buf) chars;
      Buffer.contents buf
 
let getOpenStack = fun charArr start ->
  (* For matching a "(" bracket to its corresponding ")" bracket, computes depth of stack for all characters in the array. First character reading left from our start position which has stack depth of 0 closes the bracket. *)
  let openStack = Array.make ((Array.length charArr)-start) 1 in 
    for i=1 to (Array.length openStack)-1 do
      let x = 
        if (Array.get charArr (start+i)) = '('
          then 1 
        else if (Array.get charArr (start+i)) = ')' 
          then -1
            else 0
      in
        Array.set openStack i ((Array.get openStack (i-1))+x) 
    done;
    openStack

let getClosedStack = fun charArr endS ->
  (* For matching a ")" bracket to its corresponding "(" bracket, computes depth of stack for all characters in the array. First character reading right from our endS position which has stack depth of 0 closes the bracket. *) 
  let closedStack = Array.make (endS+1) 1 in 
    for i= (endS-1) downto 0 do
      let x = 
        if (Array.get charArr (i)) = '('
          then -1 
        else if (Array.get charArr (i)) = ')' 
          then 1
            else 0
      in
        Array.set closedStack i ((Array.get closedStack (i+1))+x) 
    done;
    closedStack

let rec findZeroStart = fun intList i ->
  (* For a given openStack, finds first character reading left from our start position which has stack depth of 0 to close the "(" bracket.  *)
  if i >= Array.length intList then raise (Invalid_argument "Bracket not found.")
  else 
    if Array.get intList i =0 then i
    else findZeroStart intList (i+1)

let rec findZeroEnd = fun intList i ->
  (* For a given closedStack, finds first character reading right from our endS position which has stack depth of 0 to close the ")" bracket.  *)
  if i < 0 then raise (Invalid_argument "Bracket not found.")
  else 
    if Array.get intList i =0 then i
    else findZeroEnd intList (i-1)

let matchBracket = function 
  (* Returns index that closes the bracket in question, returns -1 if malformed input. *)
  | start, charArr, "(" -> findZeroStart (getOpenStack charArr start) start
  | endS, charArr, ")" -> findZeroEnd (getClosedStack charArr endS) endS
  | _ -> raise (Invalid_argument "Invalid input to matchBracket") 
  
let split_on_bracks = fun s -> split (regexp "(") s 
(* function for splitting string on "(" symbol to find if it contains only value and no left/right children *)

let rec trfy = fun s -> 
  (* Convert prefix equation string to a string tree *)
  if (List.length (split_on_bracks s)) >= 2
    then
      let charArr = string_to_list s |> Array.of_list in
      let subArr = Array.sub charArr 1 ((Array.length charArr)-2) in
      let rootVal = Array.sub subArr 0 1 in
      let arg1Start = 2 in
      let arg1End = matchBracket (arg1Start, subArr, "(") in
      let arg2End = (Array.length subArr)-1 in
      let arg2Start = matchBracket (arg2End, subArr, ")") in
      let leftArg = Array.sub subArr arg1Start (arg1End+1) in
      let rightArg = Array.sub subArr arg2Start (arg2End-arg2Start+1) in
      Node {value = char_array_to_string rootVal; 
            left = trfy (char_array_to_string leftArg);
            right = trfy (char_array_to_string rightArg);
      }
else
  if (List.length (split_on_bracks s)) = 1 
    then
      let charArr = string_to_list s |> Array.of_list in
      let subArr = Array.sub charArr 1 ((Array.length charArr)-2) in
      let rootVal = Array.sub subArr 0 (Array.length subArr) in 
      Node {value = char_array_to_string rootVal; 
            left = Leaf; 
            right = Leaf} 
  else
    if s = "()" || s=""
      then 
        Leaf
    else 
      raise (Invalid_argument ("Invalid input to trfy."^s))
;;

let permit_rotations = function
(* Checks if rotations are permissible or not*)
  | ("-", "-") -> (true, "-", "+")
  | ("-", "+") -> (true, "-", "-")
  | ("+", "+") -> (true, "+", "+")
  | ("+", "-") -> (true, "+", "-")
  | ("*", "*") -> (true, "*", "*")
  | ("*", "/") -> (true, "*", "/")
  | ("/", "/") -> (true, "/", "*")
  | ("/", "*") -> (true, "/", "/")
  | (a, b) -> (false, a, b)

let rrotatehelper = function
  (* Helper function to perform a right rotation on the tree *)
  | Node x -> 
    if (List.mem x.value ops) && (List.mem (getValue x.left) ops)
      then
        let (permitted, newOp1, newOp2) = permit_rotations ((getValue x.left), x.value) in
          if permitted = true then
            let originalRoot = newOp2 in
              let upperLeft = newOp1 in
                let upperRight = x.right in
                  let bottomLeft = getLeft x.left in
                  let bottomRight = getRight x.left in 
                    let rightSub = Node {value=originalRoot; left=bottomRight; right=upperRight} in
                      Node {value=upperLeft; left=bottomLeft; right=rightSub}
          else
            Node x
    else
      Node x
  | Leaf -> Leaf

let lrotatehelper = function
  (* Helper function to perform a left rotation on the tree *)
  | Node x -> 
    if (List.mem x.value ops) && (List.mem (getValue x.right) ops)
      then
      let (permitted, newOp1, newOp2) = permit_rotations (x.value, (getValue x.right)) in
      if permitted = true then
        let originalRoot = newOp1 in
          let upperLeft = x.left in
            let upperRight = newOp2 in
              let bottomLeft = getLeft x.right in
              let bottomRight = getRight x.right in 
                let leftSub = Node {value=originalRoot; left=upperLeft; right=bottomLeft} in
                  Node {value=upperRight; left=leftSub; right=bottomRight}
      else
        Node x
    else
      Node x
  | Leaf -> Leaf

let distHelper = function
  (* Helper function to perform a distribution operation on the tree *)
| Node x ->   
let leftLeft = (getLeft x.left) in 
  let leftRight = (getRight x.left) in 
    let rightLeft = (getLeft x.right) in 
      let rightRight = (getRight x.right) in 
        if ((x.value = "+" || x.value = "-")) 
          then 
          if ((getValue x.left) = "*" && (getValue x.right) = "*") || ((getValue x.left) = "/" && (getValue x.right) = "/")
            then
              if (eq (leftRight, rightRight))
                then
                  Node {value=(getValue x.left); 
                  left=Node{value=x.value; left=leftLeft; right=rightLeft};
                  right=leftRight}
              else
                if ((getValue x.left) = "*" && (getValue x.right) = "*") 
                  then 
                    if (eq (leftLeft, rightLeft))
                      then
                        Node {value=(getValue x.left); 
                        left=Node{value=x.value; left=leftRight; right=rightRight};
                        right=leftLeft}
                    else
                        if (eq (leftLeft, rightRight))
                          then
                            Node {value=(getValue x.left); 
                            left=Node{value=x.value; left=leftRight; right=rightRight};
                            right=leftLeft}
                    else
                        if (eq (leftRight, rightLeft))
                          then
                            Node {value=(getValue x.left); 
                            left=Node{value=x.value; left=leftLeft; right=rightRight};
                            right=leftRight} 
                else
                  Node x 
              else
                Node x
            else
              Node x 
        else
          Node x
| Leaf -> Leaf

let revDistHelper = function
| Leaf -> Leaf
| Node x -> 
  let leftLeft = (getLeft x.left) in 
    let leftRight = (getRight x.left) in 
      let rightLeft = (getLeft x.right) in 
        let rightRight = (getRight x.right) in 
            if (x.value = "*" || x.value = "/")
              then 
                if (getValue x.left = "+" || getValue x.left = "-")
                  then 
                    Node {value=(getValue x.left); 
                          left = Node {value=x.value; left=leftLeft; right=x.right};
                          right= Node {value=x.value; left=leftRight; right=x.right}}
            else
              if (x.value = "*") 
                then if (getValue x.right = "+" || getValue x.right = "-")
                  then 
                    Node {value=(getValue x.left); 
                          left = Node {value=x.value; left=x.left; right=rightLeft};
                          right= Node {value=x.value; left=x.left; right=rightRight}}
          else 
            Node x
        else 
          Node x 
      else 
        Node x 

let rec genSub = fun s ->
  (* Generates all possible subtrees of a tree *)
  let eqTree = trfy s in 
    if eqTree!=Leaf then
      if (getLeft eqTree) = Leaf && (getRight eqTree) = Leaf
        then [s;]
      else
        [s;] @ genSub (dtrfy (getLeft eqTree)) @ genSub (dtrfy (getRight eqTree))
    else
      []

let rec reconstruct = fun i old newT ->
  (* Reconstructs a new tree by swapping in newT in the i-th indexed subtree of old. So if subtree "k" is at the i-th index of result of genSub(old), "k" in old gets replaced by newT. *)
  let subList = genSub old in
    let oldT = trfy old in
    if (i > List.length subList)
      then oldT
    else
      if i=0 then trfy newT
      else
        let leftLength = getLength (getLeft oldT) in
        if (i <= leftLength) then 
          Node {value = getValue oldT; 
                left = reconstruct (i-1) (dtrfy (getLeft oldT)) newT; 
                right = getRight oldT}
        else
          Node {value = getValue oldT; 
                left = getLeft oldT; 
                right = reconstruct (i-1-leftLength) (dtrfy (getRight oldT)) newT}

let treeop = fun s i operation ->
  (* Perform a tree operation on tree represented by string s. *)
  let allSubs = genSub s in 
    let selectedSub = List.nth allSubs i in
      let modifiedSub = operation (trfy selectedSub) in 
        reconstruct i s (dtrfy modifiedSub)

let op = fun s x opArg -> 
  (* Performs operation on x on both sides of tree. *)
  let allSubs = genSub s in 
    let selectedSub = List.nth allSubs x in
      let eqTree = trfy s in
        let valT = getValue eqTree in 
            let leftVal = getLeft eqTree in 
              let rightVal = getRight eqTree in
                if valT = "="
                  then 
                    dtrfy( Node {value=valT;
                        left= Node {value=opArg; left=leftVal; right= (trfy selectedSub)};
                        right= Node {value=opArg; left=rightVal; right= (trfy selectedSub)};
                        }
                        )
                else 
                  s

let swapHelper = function
  (* Swaps left and right subtrees in a tree*)
  | Leaf -> Leaf
  | Node x -> 
    if ( x.value = "+" || x.value = "*" || x.value = "=" )
      then Node {value=x.value; left = x.right; right = x.left}
  else 
    if ( x.value = "-") 
      then Node {value="+"; 
      left = Node{value = x.value; left = trfy "(0)"; right = x.right}; 
      right = x.left}
  else 
    if ( x.value = "/")
      then Node {value="*"; 
      left = Node{value = x.value; left = trfy "(1)"; right = x.right}; 
      right = x.left}
    else
      Node x

let evalOp = function
  (* Determines operation z to perform on two confirmed integers x (z op) y *)
 | (x, y, "+") -> x+y
 | (x, y, "-") -> x-y
 | (x, y, "*") -> x*y
 | (x, y, "/") -> x/y 
 | _ -> raise (Invalid_argument "Invalid input to evalOp.")

let evalTree = fun x y z -> 
  (* Simplifies x (z op) y *)
    if (List.mem z ops) then
      let xVal = try Some (Option.get x) with Invalid_argument x-> None in 
        let yVal = try Some (Option.get y) with Invalid_argument y-> None in  
          if (xVal=None || yVal=None) 
            then None 
          else Option.some (evalOp ((Option.get x), (Option.get y), z))
    else 
      raise (Invalid_argument ("Invalid input to evalTree."^z))

let rec gcd a b =
  (* Computes Greatest Common Divisor of Two Integers*)
  if a<0 then 
    let c = -1*a in 
      if b = 0 then c
      else gcd b (c mod b)
  else
    if a = 0 then a
    else
      if b = 0 then a
      else gcd b (a mod b)

let div_evaluator var1 var2 leftVal rightVal rootval =
  (* Evaluates division expressions where both terms are constants *)
  if (var1 mod var2 = 0)
    then 
      Node {value = string_of_int (Option.get (evalTree leftVal rightVal rootval)); left=Leaf; right=Leaf} 
    else
      let currGCD = gcd var1 var2 in
        Node {value = rootval; 
              left=Node{value = string_of_int (var1 / currGCD); left=Leaf; right=Leaf}; 
              right= Node {value = string_of_int (var2 / currGCD); left=Leaf; right=Leaf}}
  

let rec simplifyHelper = function
  (* Simplifies an operation on two constants in a tree with root operation and constant children *)
  | Leaf -> Leaf
  | Node x -> 
    let leftSimplified = simplifyHelper x.left in
      let rightSimplified = simplifyHelper x.right in
        let leftVal = int_of_string_opt (getValue leftSimplified) in
          let rightVal = int_of_string_opt (getValue rightSimplified) in 
          if leftVal <> None &&  rightVal <> None && x.value <> "/"
            then Node {value = 
                        string_of_int (Option.get (evalTree leftVal rightVal x.value)); 
                      left=Leaf; 
                      right=Leaf}
          else
              if leftVal <> None &&  rightVal <> None && x.value = "/"
                then
                  let varA = Option.get leftVal in
                    let varB = Option.get rightVal in
                      if (varB<0)
                        then 
                          let var1 = -1 * varA in
                          let var2 = -1 * varB in
                            div_evaluator var1 var2 leftVal rightVal x.value
                        else
                          div_evaluator varA varB leftVal rightVal x.value
              else
                if ( (x.value = "+" && leftVal = Option.some 0) || ( x.value = "*" && leftVal = Option.some 1) )
                  then
                    rightSimplified          
                else if ( ( (x.value = "+" || x.value = "-") && rightVal = Option.some 0) || ( (x.value = "*" || x.value="/") && rightVal = Option.some 1) )
                  then
                    leftSimplified
                else if ((x.value = "-" && eq (leftSimplified, rightSimplified) && leftSimplified <> Leaf) || (x.value="*" && ((leftVal = Option.some 0) || (rightVal = Option.some 0))))
                  then
                    trfy "(0)"
                else if (x.value = "/" && eq (leftSimplified, rightSimplified) && leftSimplified <> Leaf)
                  then
                    trfy "(1)"
                else
                  Node{value=x.value; left = leftSimplified ; right=rightSimplified}

let _addzeroHelper = function
  (* adds 0 to expression *)
  | Leaf -> Leaf
  | Node x -> Node{value = "+"; left = Node{value=x.value; left=x.left; right=x.right}; right = trfy "(0)"}

let _subzeroHelper = function
  (* subtracts 0 from expression *)
  | Leaf -> Leaf
  | Node x -> Node{value = "-"; left = Node{value=x.value; left=x.left; right=x.right}; right = trfy "(0)"}

let _multoneHelper = function
  (* multiplies expression with 1 *)
  | Leaf -> Leaf
  | Node x -> Node{value = "*"; left = Node{value=x.value; left=x.left; right=x.right}; right = trfy "(1)"}

let _divoneHelper = function
  (* divides expression by 1 *)
  | Leaf -> Leaf
  | Node x -> Node{value = "/"; left = Node{value=x.value; left=x.left; right=x.right}; right = trfy "(1)"} 

let _newConstGen = fun a b c -> ((a*b)+c)
  (* Generates new integers from existing primitives*)

let _add = fun s x ->
  (* Adds x on both sides of the equation *)
  op s x "+"

let _sub = fun s x ->
  (* Subtracts x on both sides of the equation *)
  op s x "-"

let _mult = fun s x ->
  (* Multiplies x on both sides of the equation*)
  op s x "*"

let _div = fun s x ->
  (* Divides by x on both sides of the equation *)
  op s x "/"
  
let _rrotate = fun s i ->
  (* Performs a right rotation on i-th indexed subtree of s. *)
  dtrfy (treeop s i rrotatehelper)

let _lrotate = fun s i ->
  (* Performs a left rotation on i-th indexed subtree of s. *)
  dtrfy (treeop s i lrotatehelper)

let _swap = fun s i ->
  (* Swaps left subtree with right subtree of equation stored in i-th indexed subtree of s. *)
  dtrfy (treeop s i swapHelper)

let _simplify = fun s i ->
  (* Simplify equation stored in i-th indexed subtree of s. *)
  dtrfy (treeop s i simplifyHelper)  

let _dist = fun s i ->
  (* Implements distributive property in tree structure *)
  dtrfy (treeop s i distHelper)  

let _revdist = fun s i ->
  (* Implements reverse distributive property (A+B)*x = Ax + Bx in tree structure*)
  dtrfy (treeop s i revDistHelper)

let _addzero = fun s i ->
  (* Adds zero to a subtree *)
  dtrfy (treeop s i _addzeroHelper)

let _subzero = fun s i ->
  (* Subtracts zero from a subtree *)
  dtrfy (treeop s i _subzeroHelper)

let _multone = fun s i ->
  (* Multiplies subtree by 1 *)
  dtrfy (treeop s i _multoneHelper)

let _divone = fun s i ->
  (* Divides subtree by 1 *)
  dtrfy (treeop s i _divoneHelper)