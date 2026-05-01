# cython: language_level=3

cdef extern from "../core/chain_reaction.hpp":
    cdef struct GameState:
        signed char tokens[64]
        signed char owners[64]
        short turn_count

    void cr_init(GameState* state) nogil
    int cr_step(GameState* state, int action_idx, signed char player_id) nogil
    signed char cr_get_winner(const GameState* state) nogil


cdef class PyChainReaction:
    cdef GameState _state

    def __cinit__(self):
        cr_init(&self._state)

    def reset(self):
        cr_init(&self._state)

    def step(self, int action_idx, int player_id):
        return cr_step(&self._state, action_idx, <signed char>player_id)

    def get_winner(self):
        return cr_get_winner(&self._state)

    @property
    def tokens(self):
        return [self._state.tokens[i] for i in range(64)]

    @property
    def owners(self):
        return [self._state.owners[i] for i in range(64)]

    @property
    def turn_count(self):
        return self._state.turn_count
