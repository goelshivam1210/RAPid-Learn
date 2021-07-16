(define (domain adeGeneratedDomain)

(:requirements :typing :strips :fluents)


(:types
    actor - physical
    world - location
    material - physical
    location - object
    physical - object
    direction - object
)

(:predicates
    (crafttreetapina ?v0 - material)
    (isBlock ?v0 - physical ?v1 - location)
    (permeable ?v0 - material)
    (craftplanksina ?v0 - material)
    (crafttreetapouta ?v0 - material)
    (craftpogostickouta ?v0 - material)
    (crafttreetapinb ?v0 - material)
    (craftplanksouta ?v0 - material)
    (holding ?v0 - material)
    (crafter ?v0 - material)
    (neighbors ?v0 - world ?v1 - world)
    (unbreakable ?v0 - material)
    (orientation ?v0 - physical ?v1 - direction)
    (clockwise ?v0 - direction ?v1 - direction)
    (tapper ?v0 - material)
    (craftsticksina ?v0 - material)
    (opposite ?v0 - direction ?v1 - direction)
    (adjacent ?v0 - world ?v1 - world ?v2 - direction)
    (tapped ?v0 - location)
    (at ?v0 - physical ?v1 - location)
    (tapout ?v0 - material)
    (craftpogostickinb ?v0 - material)
    (isItem ?v0 - physical ?v1 - location)
    (craftsticksouta ?v0 - material)
    (craftpogostickinc ?v0 - material)
    (tappable ?v0 - material)
    (craftpogostickina ?v0 - material)
)

(:functions
    (inventory ?v0 - material)
)

(:action breakblock
    :parameters    (?actor - actor ?world01 - world ?direction01 - direction ?material01 - material ?world02 - world ?material02 - material )
    :precondition  (and
        (at ?actor ?world01)
        (orientation ?actor ?direction01)
        (adjacent ?world01 ?world02 ?direction01)
        (at ?material01 ?world02)
        (not (unbreakable ?material01))
        (permeable ?material02)
        (isBlock ?material01 ?world02)
    )
    :effect  (and
        (not (at ?material01 ?world02))
        (at ?material02 ?world02)
        (increase ( inventory ?material01) 1)
        (not (isBlock ?material01 ?world02))
        (isBlock ?material02 ?world02)
    )
)

(:action extractrubber
    :parameters    (?actor01 - actor ?world01 - world ?direction01 - direction ?world02 - world ?material01 - material ?world03 - world ?material02 - material )
    :precondition  (and
        (at ?actor01 ?world01)
        (orientation ?actor01 ?direction01)
        (adjacent ?world01 ?world02 ?direction01)
        (tapped ?world02)
        (tappable ?material02)
        (at ?material02 ?world03)
        (adjacent ?world02 ?world03 ?direction01)
        (tapout ?material01)
    )
    :effect  (increase ( inventory ?material01) 1)
)

(:action select
    :parameters    (?material01 - material ?material02 - material )
    :precondition  (and
        (holding ?material01)
        (>= ( inventory ?material02) 1)
    )
    :effect  (and
        (holding ?material02)
        (not (holding ?material01))
    )
)

(:action crafttreetap
    :parameters    (?material01 - material ?material02 - material ?material03 - material ?actor01 - actor ?world01 - world ?world02 - world ?direction01 - direction ?material04 - material )
    :precondition  (and
        (>= ( inventory ?material01) 1)
        (>= ( inventory ?material02) 5)
        (crafttreetapina ?material01)
        (crafttreetapinb ?material02)
        (crafttreetapouta ?material03)
        (at ?actor01 ?world01)
        (orientation ?actor01 ?direction01)
        (adjacent ?world01 ?world02 ?direction01)
        (at ?material04 ?world02)
        (crafter ?material04)
    )
    :effect  (and
        (decrease ( inventory ?material01) 1)
        (decrease ( inventory ?material02) 5)
        (increase ( inventory ?material03) 1)
    )
)

(:action turnright
    :parameters    (?actor - actor ?direction01 - direction ?direction02 - direction )
    :precondition  (and
        (orientation ?actor ?direction01)
        (clockwise ?direction01 ?direction02)
    )
    :effect  (and
        (orientation ?actor ?direction02)
        (not (orientation ?actor ?direction01))
    )
)

(:action placetreetap
    :parameters    (?actor01 - actor ?world01 - world ?world02 - world ?direction01 - direction ?world03 - world ?direction02 - direction ?material01 - material ?material02 - material ?material03 - material )
    :precondition  (and
        (at ?actor01 ?world01)
        (orientation ?actor01 ?direction01)
        (adjacent ?world01 ?world02 ?direction01)
        (adjacent ?world02 ?world03 ?direction02)
        (at ?material01 ?world02)
        (at ?material02 ?world03)
        (permeable ?material01)
        (tappable ?material02)
        (tapper ?material03)
        (>= ( inventory ?material03) 1)
    )
    :effect  (and
        (decrease ( inventory ?material03) 1)
        (at ?material03 ?world02)
        (isBlock ?material03 ?world02)
        (not (at ?material01 ?world02))
        (not (isBlock ?material01 ?world02))
        (tapped ?world02)
    )
)

(:action moveforward
    :parameters    (?actor - actor ?world01 - world ?world02 - world ?direction01 - direction ?material01 - material )
    :precondition  (and
        (at ?actor ?world01)
        (orientation ?actor ?direction01)
        (adjacent ?world01 ?world02 ?direction01)
        (at ?material01 ?world02)
        (permeable ?material01)
    )
    :effect  (and
        (at ?actor ?world02)
        (not (at ?actor ?world01))
    )
)

(:action craftpogostick
    :parameters    (?material01 - material ?material02 - material ?material03 - material ?material04 - material ?actor01 - actor ?world01 - world ?direction01 - direction ?world02 - world ?material05 - material )
    :precondition  (and
        (>= ( inventory ?material01) 4)
        (>= ( inventory ?material02) 2)
        (>= ( inventory ?material03) 1)
        (craftpogostickina ?material01)
        (craftpogostickinb ?material02)
        (craftpogostickinc ?material03)
        (craftpogostickouta ?material04)
        (at ?actor01 ?world01)
        (orientation ?actor01 ?direction01)
        (adjacent ?world01 ?world02 ?direction01)
        (at ?material05 ?world02)
        (crafter ?material05)
    )
    :effect  (and
        (decrease ( inventory ?material01) 4)
        (decrease ( inventory ?material02) 2)
        (decrease ( inventory ?material03) 1)
        (increase ( inventory ?material04) 1)
    )
)

(:action craftplanks
    :parameters    (?material01 - material ?material02 - material )
    :precondition  (and
        (>= ( inventory ?material01) 1)
        (craftplanksina ?material01)
        (craftplanksouta ?material02)
    )
    :effect  (and
        (decrease ( inventory ?material01) 1)
        (increase ( inventory ?material02) 4)
    )
)

(:action craftsticks
    :parameters    (?material01 - material ?material02 - material )
    :precondition  (and
        (>= ( inventory ?material01) 2)
        (craftsticksina ?material01)
        (craftsticksouta ?material02)
    )
    :effect  (and
        (decrease ( inventory ?material01) 2)
        (increase ( inventory ?material02) 4)
    )
)

(:action turnleft
    :parameters    (?actor - actor ?direction01 - direction ?direction02 - direction )
    :precondition  (and
        (orientation ?actor ?direction01)
        (clockwise ?direction02 ?direction01)
    )
    :effect  (and
        (orientation ?actor ?direction02)
        (not (orientation ?actor ?direction01))
    )
)

)
