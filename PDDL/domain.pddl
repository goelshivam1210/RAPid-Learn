(define (domain adeGeneratedDomain)

(:requirements :typing :strips :fluents)


(:types
    plank - breakable
    plank - placeable
    wall - physobj
    breakable - physobj
    entity - physobj
    plank - physobj
    tree_tap - craftable
    placeable - physobj
    crafting_table - physobj
    tree_tap - physobj
    plank - craftable
    coord - concept
    craftable - physobj
    crafting_table - breakable
    var - object
    crafting_table - placeable
    tapped_log - physobj
    concept - var
    rubber - physobj
    physobj - physical
    actor - physical
    pogo_stick - physobj
    stick - physobj
    tree_log - physobj
    pogo_stick - craftable
    stick - craftable
    physical - var
    tree_log - breakable
    air - physobj
    tree_log - placeable
    rubber_tree - physobj
    wooden_axe - physobj
)

(:predicates
    (holding ?v0 - physobj)
    (floating ?v0 - physobj)
    (facing ?v0 - physobj)
)

(:functions
    (world ?v0 - object)
    (inventory ?v0 - object)
    ; (totalcost)
)

(:action crafttree_tap
    :parameters    ()
    :precondition  (and
        (>= ( inventory plank) 5)
        (>= ( inventory stick) 1)
        (facing crafting_table)
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

(:action deselect
    :parameters    (?physobj01 - physobj )
    :precondition  (and
        (>= ( inventory ?physobj01) 1)
        (holding ?physobj01)
    )
    :effect  (and
        (not (holding ?physobj01))
        (holding air)
    )
)

(:action pickup
    :parameters    (?physobj01 - physobj ?physobj02 - physobj )
    :precondition  (and
        (floating ?physobj01)
        (facing ?physobj02)
    )
    :effect  (and
        (not (floating ?physobj01))
        (not (facing ?physobj01))
        (not (facing ?physobj02))
        (facing air)
        (increase ( inventory ?physobj01) 1)
        (decrease ( world ?physobj01) 1)
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

(:action place
    :parameters    (?placeable01 - placeable )
    :precondition  (and
        (facing air)
        (>= ( inventory ?placeable01) 1)
    )
    :effect  (and
        (facing ?placeable01)
        (not (facing air))
        (increase ( world ?placeable01) 1)
        (decrease ( inventory ?placeable01) 1)
        (decrease ( world air) 1)
    )
)

(:action break
    :parameters    (?breakable01 - breakable )
    :precondition  (and
        (facing ?breakable01)
        (not (floating ?breakable01))
    )
    :effect  (and
        (facing air)
        (not (facing ?breakable01))
        (increase ( inventory ?breakable01) 1)
        (increase ( world air) 1)
        (decrease ( world ?breakable01) 1)
    )
)

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
        ; (not (facing air))
        ; (facing tree_tap)
        (increase ( inventory rubber) 1)
        ; (decrease ( inventory tree_tap) 1)
        ; (decrease ( world air) 1)
        ; (increase ( world tree_tap) 1)
    )
)

; (:action placetreetap
;     :parameters    ()
;     :precondition  (and
;         (>= ( inventory tree_tap) 1)
;         (facing air)
;         ()
;     )
;     :effect  (and
;         (facing tree_tap)
;         (decrease ( inventory tree_tap) 1)
;         (decrease ( world air) 1)
;         (increase ( world tree_tap) 1)
;     )
; )

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
    :parameters    (?physobj01 - physobj )
    :precondition  (and
        (>= ( inventory ?physobj01) 1)
        (holding air)
    )
    :effect  (and
        (holding ?physobj01)
        (not (holding air))
    )
)

)