(define (domain adeGeneratedDomain)

(:requirements :typing :strips :fluents)


(:types
	wooden_axe - physobj
	water - physobj
	rubber_tree - physobj
    ; plank - placeable
    wall - physobj
    entity - physobj
    plank - physobj
    ; tree_tap - craftable
    ; placeable - physobj
    crafting_table - physobj
    tree_tap - physobj
    ; plank - craftable
    ; coord - concept
    ; craftable - physobj
    ; var - object
    ; crafting_table - placeable
    tapped_log - physobj
    ; concept - var
    rubber - physobj
    physobj - physical
    actor - physical
    pogo_stick - physobj
    stick - physobj
    tree_log - physobj
    ; pogo_stick - craftable
    ; stick - craftable
    ; physical - var
    air - physobj
    ; tree_log - placeable
    ; water - physobj
)

(:predicates
    (holding ?v0 - physobj)
    (floating ?v0 - physobj)
    (facing ?v0 - physobj)
    (onfire ?v0 - physobj)
)

(:functions
    (world ?v0 - object)
    (inventory ?v0 - object)
    ; (totalcost)
)

(:action spray
    :parameters    (?physobj01 - physobj)
    :precondition (and  
            (facing ?physobj01)
            (onfire ?physobj01)
            (holding water)
    )
    :effect (not(onfire crafting_table))
)

(:action spray_crafting_table
    :parameters    ()
    :precondition (and  
            (facing crafting_table)
            (onfire crafting_table)
            (holding water)
    )
    :effect (not(onfire crafting_table))
)

(:action crafttree_tap
    :parameters    ()
    :precondition  (and
        (>= ( inventory plank) 4)
        (>= ( inventory stick) 1)
        (facing crafting_table)
        (not (onfire crafting_table))
    )
    :effect  (and
        (increase ( inventory tree_tap) 1)
        (decrease ( inventory plank) 5)
        (decrease ( inventory stick) 1)
    )
)

(:action approach
    :parameters    (?physobj01 - physobj ?physobj02 - physobj )
    :precondition  (and
        (>= ( world ?physobj02) 1)
        (facing ?physobj01)
    )
    :effect  (and
        (facing ?physobj02)
        (not (facing ?physobj01))
    )
)

; (:action approach_floating
;     :parameters    (?physobj01 - physobj ?physobj02 - physobj )
;     :precondition  (and
;         (>= ( world ?physobj02) 1)
;         (facing ?physobj01)
;     )
;     :effect  (and
;         (facing ?physobj02)
;         (not (facing ?physobj01))
;     )
; )

(:action approach_floating
    :parameters    (?physobj01 - physobj ?physobj02 - physobj)
    :precondition  (and
        (>= ( world ?physobj02) 1)
        (facing ?physobj01)
        (floating ?physobj02)
    )
    :effect  (and
        (facing air)
        (decrease ( world ?physobj02) 1)
        (increase ( inventory ?physobj02) 1)
        (not (floating ?physobj02))
        (increase ( world air) 1)

    )
)

(:action craftplank
    :parameters    ()
    :precondition  (>= ( inventory tree_log) 1)
    :effect  (and
        (increase ( inventory plank) 4)
        (decrease ( inventory tree_log) 1)
    )
)

(:action break_holding
    :parameters    ()
    :precondition  (and
        (facing tree_log)
        (not (floating tree_log))
        (holding wooden_axe)
    )
    :effect  (and
        (facing air)
        (not (facing tree_log))
        (increase ( inventory tree_log) 1)
        (increase ( world air) 1)
        (decrease ( world tree_log) 1)
    )
)

; (:action break
;     :parameters    ()
;     :precondition  (and
;         (facing tree_log)
;         (not (floating tree_log))
;     )
;     :effect  (and
;         (facing air)
;         (not (facing tree_log))
;         (increase ( inventory tree_log) 1)
;         (increase ( world air) 1)
;         (decrease ( world tree_log) 1)
;     )
; )

(:action craftstick
    :parameters    ()
    :precondition  (>= ( inventory plank) 2)
    :effect  (and
        (increase ( inventory stick) 4)
        (decrease ( inventory plank) 2)
    )
)

(:action extractrubber
    :parameters    ()
    :precondition  (and
        (>= ( inventory tree_tap) 1)
        (facing tree_log)
        (holding tree_tap)
    )
    :effect  (and
        (increase ( inventory rubber) 1)
    )
)

(:action craftpogo_stick
    :parameters    ()
    :precondition  (and
        (>= ( inventory plank) 2)
        (>= ( inventory stick) 4)
        (>= ( inventory rubber) 1)
        (facing crafting_table)
    )
    :effect  (and
        (increase ( inventory pogo_stick) 1)
        (decrease ( inventory plank) 2)
        (decrease ( inventory stick) 4)
        (decrease ( inventory rubber) 1)
    )
)

(:action select
    :parameters    (?physobj01 - physobj ?physobj02 - physobj)
    :precondition  (and
        (>= ( inventory ?physobj01) 1)
        (>= ( inventory ?physobj02) 1)
        (not (holding ?physobj02))
        (holding ?physobj01)
    )
    :effect  (and
        (not(holding ?physobj01))
        (holding ?physobj02)
    )
)

; (:action deselect
;     :parameters    (?physobj01 - physobj )
;     :precondition  (and
;         (>= ( inventory ?physobj01) 1)
;         (holding ?physobj01)
;     )
;     :effect  (and
;         (holding air)
;         (not (holding ?physobj01))
;     )
; )
)