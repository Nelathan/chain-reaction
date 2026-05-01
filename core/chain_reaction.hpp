#pragma once
#include <stdint.h>

#define CR_WIDTH 8
#define CR_HEIGHT 8
#define CR_CELLS 64

struct GameState {
    int8_t tokens[CR_CELLS];
    int8_t owners[CR_CELLS];
    int16_t turn_count;
};

inline int8_t cr_get_mass(int idx) {
    int x = idx % CR_WIDTH;
    int y = idx / CR_WIDTH;
    int8_t mass = 4;
    if (x == 0 || x == CR_WIDTH - 1) mass--;
    if (y == 0 || y == CR_HEIGHT - 1) mass--;
    return mass;
}

inline void cr_zero_state(GameState* state) {
    for (int i = 0; i < CR_CELLS; ++i) {
        state->tokens[i] = 0;
        state->owners[i] = 0;
    }
    state->turn_count = 0;
}

inline void cr_copy_cells(int8_t* dst, const int8_t* src) {
    for (int i = 0; i < CR_CELLS; ++i) {
        dst[i] = src[i];
    }
}

inline int cr_is_legal_move(const GameState* state, int action_idx, int8_t player_id) {
    if (player_id != 1 && player_id != 2) return 0;
    if (action_idx < 0 || action_idx >= CR_CELLS) return 0;
    return state->owners[action_idx] == 0 || state->owners[action_idx] == player_id;
}

inline void cr_scatter(int8_t* next_tokens, int8_t* next_owners, int idx, int8_t owner) {
    next_tokens[idx]++;
    next_owners[idx] = owner;
}

inline int cr_step(GameState* state, int action_idx, int8_t player_id) {
    if (!cr_is_legal_move(state, action_idx, player_id)) return 0;

    state->tokens[action_idx]++;
    state->owners[action_idx] = player_id;
    state->turn_count++;

    int8_t next_tokens[CR_CELLS];
    int8_t next_owners[CR_CELLS];

    int unstable = 1;
    while (unstable) {
        unstable = 0;

        cr_copy_cells(next_tokens, state->tokens);
        cr_copy_cells(next_owners, state->owners);

        for (int i = 0; i < CR_CELLS; ++i) {
            int8_t mass = cr_get_mass(i);
            if (state->tokens[i] >= mass) {
                unstable = 1;
                int8_t owner = state->owners[i];

                next_tokens[i] -= mass;
                if (next_tokens[i] == 0) {
                    next_owners[i] = 0;
                }

                int x = i % CR_WIDTH;
                int y = i / CR_WIDTH;

                /* Opposing simultaneous scatters resolve deterministically by scan order. */
                if (y > 0) cr_scatter(next_tokens, next_owners, i - CR_WIDTH, owner);
                if (y < CR_HEIGHT - 1) cr_scatter(next_tokens, next_owners, i + CR_WIDTH, owner);
                if (x > 0) cr_scatter(next_tokens, next_owners, i - 1, owner);
                if (x < CR_WIDTH - 1) cr_scatter(next_tokens, next_owners, i + 1, owner);
            }
        }

        cr_copy_cells(state->tokens, next_tokens);
        cr_copy_cells(state->owners, next_owners);
    }

    return 1;
}

inline int8_t cr_get_winner(const GameState* state) {
    if (state->turn_count < 2) return 0;

    int p1 = 0;
    int p2 = 0;
    for (int i = 0; i < CR_CELLS; ++i) {
        if (state->owners[i] == 1) p1 = 1;
        if (state->owners[i] == 2) p2 = 1;
        if (p1 && p2) return 0;
    }
    if (p1) return 1;
    if (p2) return 2;
    return 0;
}

inline void cr_init(GameState* state) {
    cr_zero_state(state);
}
