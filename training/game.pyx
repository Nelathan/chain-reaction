cdef extern from "../core-logic/chain_reaction.hpp":
    struct Cell:
        char count
        char owner

    struct GameState:
        Cell cells[64]

    bool step(GameState* s, int action_index, int player_id)

# PufferLib will instantiate thousands of these
cdef class PyGameState:
    cdef GameState state

    def apply_action(self, int action_index, int player_id):
        return step(&self.state, action_index, player_id)
