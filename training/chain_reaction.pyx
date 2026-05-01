# cython: language_level=3

cdef extern from "../core/chain_reaction.hpp":
    cdef struct GameState:
        signed char tokens[64]
        signed char owners[64]
        short turn_count
        signed char players_alive_mask
        signed char last_move_exploded

    cdef struct WaveLog:
        short wave_count
        signed char wave_log_truncated
        signed char wave_exploded[4096]
        signed char wave_tokens[4096]
        signed char wave_owners[4096]

    void cr_init(GameState* state) nogil
    void cr_init_wave_log(WaveLog* log) nogil
    int cr_step(GameState* state, int action_idx, signed char player_id) nogil
    int cr_step_with_log(GameState* state, WaveLog* log, int action_idx, signed char player_id) nogil
    signed char cr_get_winner(const GameState* state) nogil
    void cr_write_legal_actions(const GameState* state, signed char player_id, signed char* out_actions) nogil
    void cr_write_observation(const GameState* state, signed char player_id, signed char* out_observation) nogil


cdef class PyChainReaction:
    cdef GameState _state
    cdef WaveLog _wave_log

    def __cinit__(self):
        cr_init(&self._state)
        cr_init_wave_log(&self._wave_log)

    def reset(self):
        cr_init(&self._state)
        cr_init_wave_log(&self._wave_log)

    def step(self, int action_idx, int player_id):
        return cr_step_with_log(&self._state, &self._wave_log, action_idx, <signed char>player_id)

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

    @property
    def wave_count(self):
        return self._wave_log.wave_count

    @property
    def wave_log_truncated(self):
        return self._wave_log.wave_log_truncated

    @property
    def wave_exploded(self):
        return [self._wave_log.wave_exploded[i] for i in range(4096)]

    @property
    def wave_tokens(self):
        return [self._wave_log.wave_tokens[i] for i in range(4096)]

    @property
    def wave_owners(self):
        return [self._wave_log.wave_owners[i] for i in range(4096)]
