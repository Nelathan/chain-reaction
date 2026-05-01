# cython: language_level=3

cdef extern from "../core/chain_reaction.hpp":
    cdef struct GameState:
        signed char tokens[64]
        signed char owners[64]
        short turn_count
        signed char players_alive_mask
        signed char last_move_exploded

    void cr_init(GameState* state) nogil
    int cr_step(GameState* state, int action_idx, signed char player_id) nogil
    signed char cr_get_winner(const GameState* state) nogil
    void cr_write_legal_actions(const GameState* state, signed char player_id, signed char* out_actions) nogil
    void cr_write_observation(const GameState* state, signed char player_id, signed char* out_observation) nogil


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

    def legal_actions(self, int player_id):
        cdef signed char actions[64]
        cr_write_legal_actions(&self._state, <signed char>player_id, actions)
        return [actions[i] for i in range(64)]

    def observation(self, int player_id):
        cdef signed char values[64]
        cr_write_observation(&self._state, <signed char>player_id, values)
        return [values[i] for i in range(64)]

    @property
    def tokens(self):
        return [self._state.tokens[i] for i in range(64)]

    @property
    def owners(self):
        return [self._state.owners[i] for i in range(64)]

    @property
    def turn_count(self):
        return self._state.turn_count

    @property
    def players_alive_mask(self):
        return self._state.players_alive_mask

    @property
    def last_move_exploded(self):
        return self._state.last_move_exploded
