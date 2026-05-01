#pragma once
#include <stdint.h>

struct Cell {
    uint8_t count;
    uint8_t owner;
};

struct GameState {
    Cell cells[64];
    Cell temp_cells[64];
    int width = 8;
    int height = 8;
};

// Pure math. No allocations. No standard library.
bool step(GameState* state, int action_index, int player_id);
int get_winner(GameState* state);
