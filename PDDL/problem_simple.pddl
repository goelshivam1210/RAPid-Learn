(define (problem adeGeneratedProblem)

(:domain adeGeneratedDomain)

(:objects
    crafting_table - crafting_table
    tree_tap - tree_tap
    tree_log - tree_log
    rubber - rubber
    pogo_stick - pogo_stick
    stick - stick
    air - air
    wall - wall
    plank - plank
)

(:init
    (= ( world stick) 0)
    (= ( inventory wall) 0)
    (= ( inventory stick) 0)
    (= ( world wall) 124)
    (= ( inventory crafting_table) 0)
    (= ( world crafting_table) 1)
    (= ( inventory pogo_stick) 0)
    (= ( inventory tree_log) 0)
    (= ( world air) 897)
    (= ( inventory plank) 0)
    (= ( inventory rubber) 0)
    (= ( world rubber) 0)
    (= ( inventory air) 0)
    (= ( world tree_tap) 0)
    (= ( inventory tree_tap) 0)
    (= ( world tree_log) 4)
    (near crafting_table)
    (= ( world pogo_stick) 0)
    (= ( world plank) 0))

(:goal (>= ( inventory pogo_stick) 1)
)

)