# Mondrian Solver

A state-space implementation of the [Mondrian tiling problem](https://www.youtube.com/watch?v=49KvZrioFB0).

The state space is explored optimally using a Breadth-First Search implementation for a starting square of size `a` and a depth of `M`.

Actions upon states are defined as either merging or splitting of (an) exisiting rectangle(s). For splitting operations, this can occur either horizontally or vertically at any index across the rectangle's height and width respectively.

The state-space is refined by not allowing any states that are congruent - either rotated or flipped versions of existing states - defined using a hashing function. This helps to decrease the state-space while searching, however, with a greater `M`, the time complexitiy is still exponential due to Breadth-First search being used.

Example results:

| `M`v   a > | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |