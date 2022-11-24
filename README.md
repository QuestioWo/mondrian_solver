# Mondrian Solver

A state-space implementation of the [Mondrian tiling problem](https://www.youtube.com/watch?v=49KvZrioFB0).

The state space is explored optimally using a [Breadth-First Search (BFS)](https://en.wikipedia.org/wiki/Breadth-first_search) or an [Iterative-Deepening Depth-First Search (IDDFS)](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search) implementation for a starting square of size `a`x`a` and a state-space depth of `M`.

Actions upon states are defined as either merging or splitting of (an) exisiting rectangle(s). For splitting operations, this can occur either horizontally or vertically at any index across the rectangle's height and width respectively.

The state-space is refined by not allowing any states that are congruent - either rotated or flipped versions of existing states - defined using a hashing function. This helps to decrease the state-space while searching, however, with a greater `M`, the time complexitiy is still exponential due to the searching methods being used.

Originally, only BFS was implemented, however, due to its large memory requirements, IDDFS was implemented. IDDFS allows for the same complete-ness of BFS, but at the memory cost of a Depth-First Search style implementation - ~O(M) vs O(a^M) for BFS.

The currently code include a parallelised BFS implementation (using python's `multiprocessing` library) and a non-parallelised, however, memory-light IDDFS implementation. The IDDFS implementation is currently used by default, however, the BFS implementation can be used by switching the function called inside of the `SolveMondrian` function. 

Both the BFS and IDDFS implementations have the same time complexitiy. However, the BFS implementation is faster due to its parallelisation. This makes it the superior option for smaller state-spaces, however not so for larger due to its memory requirements - where the slower IDDFS implementation has to beused instead.

## How to run

Due to the use of a `match` statement, this repository can only be ran with Python >3.10. With a Python >3.10 installation, the repository can be ran by running:

```bash
python3.10 -m pip install virtualenv -U

virtualenv venv -p=$(which python3.10)
source venv/bin/acivate

python -m pip install -r requirements.txt
python main.py
```

The `a` and `M` values can be changed by changing the `a_s` and `M_s` lists inside of `def main()`.

Images of the (found) optimal Mondrian patterns will be saved as `.png` files labelled as `best_aXM=score.png`. Additionally, the script will print the compute times and (found) optimal scores for each `a_s` and `M_s` combination.

## Example results

| `M`v   `a`> | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|
| 2 | ![](assets/best_3X2%3D2.png) | ![](assets/best_4X2%3D6.png) | ![](assets/best_5X2%3D4.png) | ![](assets/best_6X2%3D8.png) | ![](assets/best_7X2%3D6.png) |
| 3 | ![](assets/best_3X3%3D2.png) | ![](assets/best_4X3%3D4.png) | ![](assets/best_5X3%3D4.png) | ![](assets/best_6X3%3D6.png) | ![](assets/best_7X3%3D6.png) |
| 4 | ![](assets/best_3X4%3D2.png) | ![](assets/best_4X4%3D4.png) | ![](assets/best_5X4%3D4.png) | ![](assets/best_6X4%3D5.png) | ![](assets/best_7X4%3D5.png) |
| 5 | ![](assets/best_3X5%3D2.png) | ![](assets/best_4X5%3D4.png) | ![](assets/best_5X5%3D4.png) | ![](assets/best_6X5%3D5.png) | ![](assets/best_7X5%3D5.png) |
| 6 | ![](assets/best_3X6%3D2.png) | ![](assets/best_4X6%3D4.png) | ![](assets/best_5X6%3D4.png) | ![](assets/best_6X6%3D5.png) | ![](assets/best_7X6%3D5.png) |


| `M`v   `a`> | 8 | 12 | 16 | 20 |
|---|---|---|---|---|
| 2 | ![](assets/best_8X2%3D10.png) | ![](assets/best_12X2%3D16.png) | ![](assets/best_16X2%3D22.png) | ![](assets/best_20X2%3D26.png) |
| 3 | ![](assets/best_8X3%3D10.png) | ![](assets/best_12X3%3D8.png) | ![](assets/best_16X3%3D22.png) | ![](assets/best_20X3%3D14.png) |
| 4 | ![](assets/best_8X4%3D9.png) | ![](assets/best_12X4%3D8.png) | ![](assets/best_16X4%3D18.png) | ![](assets/best_20X4%3D13.png) |
| 5 | ![](assets/best_8X5%3D6.png) | ![](assets/best_12X5%3D8.png) | ![](assets/best_16X5%3D17.png) | ![](assets/best_20X5%3D13.png) |