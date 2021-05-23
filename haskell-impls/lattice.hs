module Main where
-- import Control.Parallel.Strategies (parMap, rpar)

-- 3xN Lattice Sprague-Grundy Calculations

pickCoords n = sequence [[0,1,2], [0..n-1]]
mex list = head (filter (`notElem` list) [0..(maximum list+1)])
checkAdj [x,y] [n,m] = not ((x==n && abs(y-m) <= 1) || (y==m && abs (x-n) <= 1))
nextMoves max history = filter (\move -> null history || all (checkAdj move) history) (pickCoords max)
calcNimber max history | null (nextMoves max history) = 0 | otherwise = mex (map (\move -> calcNimber max (history ++ [move])) (nextMoves max history))
a316632 n = calcNimber n []

{-|
-- Parallel threaded version (no performance speedup rn)
parCalcNimber :: Int -> [[Int]] -> Int
parCalcNimber max history | null (nextMoves max history) = 0 | otherwise = mex (parMap rpar (\move -> calcNimber max (history ++ [move])) (nextMoves max history))

parA316632:: Int -> Int
parA316632 = parCalcNimber n []
-}

-- Example usage (to find a(4))
main = print (a316632 4)
