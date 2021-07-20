(define
	(problem adeGeneratedProblem)
	(:domain adeGeneratedDomain)
	(:objects
		crafting_table - crafting_table
		plank - plank
		tree_log - tree_log
		tree_tap - tree_tap
		pogo_stick - pogo_stick
		stick - stick
		rubber - rubber
		air - air
		wall - wall
	)
	(:init
		(= (inventory crafting_table) 0)
		(= (world crafting_table) 1)
		(= (inventory plank) 0)
		(= (world plank) 0)
		(= (inventory tree_log) 0)
		(= (world tree_log) 5)
		(= (inventory tree_tap) 0)
		(= (world tree_tap) 0)
		(= (inventory pogo_stick) 0)
		(= (world pogo_stick) 0)
		(= (inventory stick) 0)
		(= (world stick) 0)
		(= (inventory rubber) 0)
		(= (world rubber) 0)
		(= (inventory air) 0)
		(= (world air) 58)
		(= (inventory wall) 0)
		(= (world wall) 36)
		(holding air)
		(near crafting_table)
	)
(:goal (>= (inventory pogo_stick) 1))
)
