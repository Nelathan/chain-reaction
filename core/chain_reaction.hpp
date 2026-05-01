#pragma once
#include <stdint.h>

#define CR_WIDTH 8
#define CR_HEIGHT 8
#define CR_CELLS 64
#define CR_PLAYERS 2
#define CR_MAX_LOGGED_WAVES 64

struct GameState {
    int8_t tokens[CR_CELLS];
    int8_t owners[CR_CELLS];
    int16_t turn_count;
    int8_t players_alive_mask;
    int8_t last_move_exploded;
};

struct WaveLog {
    int16_t wave_count;
    int8_t wave_log_truncated;
    int8_t wave_exploded[CR_MAX_LOGGED_WAVES * CR_CELLS];
    int8_t wave_tokens[CR_MAX_LOGGED_WAVES * CR_CELLS];
    int8_t wave_owners[CR_MAX_LOGGED_WAVES * CR_CELLS];
};

inline int8_t cr_player_bit(int8_t player_id) {
    return (int8_t)(1 << (player_id - 1));
}

inline int8_t cr_all_players_mask(void) {
    return (int8_t)((1 << CR_PLAYERS) - 1);
}

inline int cr_count_bits(int8_t mask) {
    int count = 0;
    uint8_t value = (uint8_t)mask;
    while (value != 0) {
        count += value & 1;
        value >>= 1;
    }
    return count;
}

inline int8_t cr_mask_to_player(int8_t mask) {
    for (int8_t player = 1; player <= CR_PLAYERS; ++player) {
        if (mask == cr_player_bit(player)) return player;
    }
    return 0;
}

inline int8_t cr_get_mass(int idx) {
    int x = idx % CR_WIDTH;
    int y = idx / CR_WIDTH;
    int8_t mass = 4;
    if (x == 0 || x == CR_WIDTH - 1) mass--;
    if (y == 0 || y == CR_HEIGHT - 1) mass--;
    return mass;
}

inline void cr_init(GameState* state) {
    for (int i = 0; i < CR_CELLS; ++i) {
        state->tokens[i] = 0;
        state->owners[i] = 0;
    }
    state->turn_count = 0;
    state->players_alive_mask = cr_all_players_mask();
    state->last_move_exploded = 0;
}

inline void cr_copy_cells(int8_t* dst, const int8_t* src) {
    for (int i = 0; i < CR_CELLS; ++i) {
        dst[i] = src[i];
    }
}

inline void cr_zero_cells(int8_t* cells) {
    for (int i = 0; i < CR_CELLS; ++i) {
        cells[i] = 0;
    }
}

inline void cr_init_wave_log(WaveLog* log) {
    log->wave_count = 0;
    log->wave_log_truncated = 0;
    for (int i = 0; i < CR_MAX_LOGGED_WAVES * CR_CELLS; ++i) {
        log->wave_exploded[i] = 0;
        log->wave_tokens[i] = 0;
        log->wave_owners[i] = 0;
    }
}

inline int cr_is_legal_move(const GameState* state, int action_idx, int8_t player_id) {
    if (player_id < 1 || player_id > CR_PLAYERS) return 0;
    if (action_idx < 0 || action_idx >= CR_CELLS) return 0;
    if (cr_count_bits(state->players_alive_mask) == 1) return 0;
    if ((state->players_alive_mask & cr_player_bit(player_id)) == 0) return 0;
    return state->owners[action_idx] == 0 || state->owners[action_idx] == player_id;
}

inline void cr_add_pressure(int8_t* pressure, int idx, int8_t owner) {
    pressure[(owner - 1) * CR_CELLS + idx]++;
}

inline void cr_apply_pressure(int8_t* next_tokens, int8_t* next_owners, const int8_t* pressure) {
    for (int i = 0; i < CR_CELLS; ++i) {
        int p1 = pressure[i];
        int p2 = pressure[CR_CELLS + i];
        int net = p1 - p2;

        if (net > 0) {
            next_tokens[i] = (int8_t)(next_tokens[i] + net);
            next_owners[i] = 1;
        } else if (net < 0) {
            next_tokens[i] = (int8_t)(next_tokens[i] - net);
            next_owners[i] = 2;
        } else if (next_tokens[i] == 0) {
            next_owners[i] = 0;
        }
    }
}

inline void cr_log_wave(const GameState* state, WaveLog* log, const int8_t* exploded_cells) {
    if (log == 0) return;
    if (log->wave_count >= CR_MAX_LOGGED_WAVES) {
        log->wave_log_truncated = 1;
        return;
    }

    int base = log->wave_count * CR_CELLS;
    for (int i = 0; i < CR_CELLS; ++i) {
        log->wave_exploded[base + i] = exploded_cells[i];
        log->wave_tokens[base + i] = state->tokens[i];
        log->wave_owners[base + i] = state->owners[i];
    }
    log->wave_count++;
}

inline int cr_resolve_wave(GameState* state, WaveLog* log, int8_t* next_tokens, int8_t* next_owners, int8_t* pressure, int8_t* exploded_cells) {
    int exploded = 0;

    cr_copy_cells(next_tokens, state->tokens);
    cr_copy_cells(next_owners, state->owners);
    cr_zero_cells(pressure);
    cr_zero_cells(pressure + CR_CELLS);
    cr_zero_cells(exploded_cells);

    for (int i = 0; i < CR_CELLS; ++i) {
        int8_t owner = state->owners[i];
        if (owner == 0) continue;

        int8_t mass = cr_get_mass(i);
        if (state->tokens[i] >= mass) {
            exploded = 1;
            exploded_cells[i] = 1;
            next_tokens[i] = (int8_t)(next_tokens[i] - mass);
            if (next_tokens[i] == 0) {
                next_owners[i] = 0;
            }

            int x = i % CR_WIDTH;
            int y = i / CR_WIDTH;

            if (y > 0) cr_add_pressure(pressure, i - CR_WIDTH, owner);
            if (y < CR_HEIGHT - 1) cr_add_pressure(pressure, i + CR_WIDTH, owner);
            if (x > 0) cr_add_pressure(pressure, i - 1, owner);
            if (x < CR_WIDTH - 1) cr_add_pressure(pressure, i + 1, owner);
        }
    }

    if (!exploded) return 0;

    cr_apply_pressure(next_tokens, next_owners, pressure);
    cr_copy_cells(state->tokens, next_tokens);
    cr_copy_cells(state->owners, next_owners);
    cr_log_wave(state, log, exploded_cells);
    return 1;
}

inline void cr_update_alive_mask(GameState* state) {
    int8_t alive = 0;
    for (int i = 0; i < CR_CELLS; ++i) {
        int8_t owner = state->owners[i];
        if (owner >= 1 && owner <= CR_PLAYERS) alive |= cr_player_bit(owner);
    }

    state->players_alive_mask = alive;
}

inline int cr_step_internal(GameState* state, WaveLog* log, int action_idx, int8_t player_id) {
    if (!cr_is_legal_move(state, action_idx, player_id)) return 0;

    state->tokens[action_idx]++;
    state->owners[action_idx] = player_id;
    state->turn_count++;
    state->last_move_exploded = 0;
    if (log != 0) cr_init_wave_log(log);

    int8_t next_tokens[CR_CELLS];
    int8_t next_owners[CR_CELLS];
    int8_t pressure[CR_PLAYERS * CR_CELLS];
    int8_t exploded_cells[CR_CELLS];

    while (cr_resolve_wave(state, log, next_tokens, next_owners, pressure, exploded_cells)) {
        state->last_move_exploded = 1;
    }

    if (state->last_move_exploded != 0) cr_update_alive_mask(state);
    return 1;
}

inline int cr_step(GameState* state, int action_idx, int8_t player_id) {
    return cr_step_internal(state, 0, action_idx, player_id);
}

inline int cr_step_with_log(GameState* state, WaveLog* log, int action_idx, int8_t player_id) {
    return cr_step_internal(state, log, action_idx, player_id);
}

inline int8_t cr_get_winner(const GameState* state) {
    if (cr_count_bits(state->players_alive_mask) == 1) return cr_mask_to_player(state->players_alive_mask);
    return 0;
}

inline void cr_write_legal_actions(const GameState* state, int8_t player_id, int8_t* out_actions) {
    for (int i = 0; i < CR_CELLS; ++i) {
        out_actions[i] = cr_is_legal_move(state, i, player_id) ? 1 : 0;
    }
}

inline void cr_write_observation(const GameState* state, int8_t player_id, int8_t* out_observation) {
    for (int i = 0; i < CR_CELLS; ++i) {
        int8_t owner = state->owners[i];
        if (owner == 0) {
            out_observation[i] = 0;
        } else if (owner == player_id) {
            out_observation[i] = (int8_t)(state->tokens[i] - cr_get_mass(i));
        } else {
            out_observation[i] = (int8_t)((state->tokens[i] - cr_get_mass(i)) * -1);
        }
    }
}
