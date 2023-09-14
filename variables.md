

# Variable node in AST

 - De Brujin notation for simplicity
 - All variables should be typed in the AST internally.
 - Perhaps we can have a reader that reads variables and computes their types.

 - Something like `(var_[i] $2)` is used to refer to `$2` but with guaranteed type `[i]`

# Lambda node in AST

 - Lambda is also typed in the AST
 - Something like `(lambda_{f, 3} (+ (var_{f, 3} $1) 2))` is used to represent
    a function that takes in `{f, 3}` and returns `{f, 3}` (return type should
    be computable so doesn't need to be in the AST)
 - We do *not* support polymorphism, that seems like far too much work
    - is this ok for dreamcoder??

# Enumeration/search

 - 