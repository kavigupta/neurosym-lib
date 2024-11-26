
type Symbol = String

hNear dslHL cost (refineMap :: Map Symbol DSL) = do
    p_HL <- near dslHL cost
    let ip_HL = freeze (initialize p_HL)
    p_concrete <- iterateUntilM' (not . hasSymbol (keys refineMap)) ip_HL $ \ip -> do
        let node = withSymbol (keys refineMap) ip
        let holeDSL = refineMap ! symbol node
        let cost' fp = cost $ substitute ip node fp
        p_node <- near holeDSL cost'
        let ip_node = initialize p_node
        return $ substitute ip node (freeze ip_node)
    return p_concrete


(>>=) :: SearchGraph a -> (a -> SearchGraph b) -> SearchGraph b