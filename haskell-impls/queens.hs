module Main where
-- import Control.Parallel.Strategies (parMap, rpar)

-- NxN Chessboard Sprague-Grundy Calculations for Queens

pickCoords n = sequence (replicate 2 [0..n-1])
mex list = head (filter (`notElem` list) [0..(maximum list+1)])
checkIntersect [x,y] [n,m] = not (x == n || y == m) && (abs (x-n) /= abs (y-m))
nextMoves max history = filter (\move -> null history || all (checkIntersect move) history) (pickCoords max)
calcNimber max history | null (nextMoves max history) = 0 | otherwise = mex (map (\move -> calcNimber max (history ++ [move])) (nextMoves max history))
a344227 n = calcNimber n []

{-|
-- Parallel threaded version (no performance speedup rn)
parCalcNimber :: Int -> [[Int]] -> Int
parCalcNimber max history | null (nextMoves max history) = 0 | otherwise = mex (parMap rpar (\move -> calcNimber max (history ++ [move])) (nextMoves max history))

parA344227 :: Int -> Int
parA344227 n = parCalcNimber n []
-}

-- Example usage (to find a(4))
main = print (a344227 4)
