#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"
typedef struct GameState GameState;
typedef struct WaveLog WaveLog;
#define inline static inline
#include "chain_reaction_core/chain_reaction.hpp"
#undef inline

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
    float illegal_moves;
    float truncations;
    float terminal_games;
    float terminal_rewards;
    float cascade_depth;
    float cascade_events;
    float max_cascade_depth;
    float cascade_log_truncations;
    float p1_wins;
    float p1_games;
} Log;

typedef struct {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    int max_turns;
    int current_player;
    unsigned int rng;
    int active_width;
    int active_height;
    GameState state;
} ChainReactionOcean;

static inline void cr_ocean_write_observation(ChainReactionOcean* env) {
    int8_t obs[CR_CELLS];
    cr_write_observation(&env->state, (int8_t)env->current_player, obs);
    for (int i = 0; i < CR_CELLS; ++i) {
        env->observations[i] = (float)obs[i];
    }
}

static inline void cr_ocean_add_game_log(ChainReactionOcean* env, float p1_result, float actor_reward, int is_terminal, int is_truncation) {
    env->log.score += p1_result;
    env->log.episode_return += p1_result;
    env->log.episode_length += (float)env->state.turn_count;
    env->log.perf += p1_result > 0.0f ? 1.0f : 0.0f;
    env->log.n += 1.0f;
    env->log.p1_games += 1.0f;
    if (p1_result > 0.0f) env->log.p1_wins += 1.0f;
    if (is_terminal) {
        env->log.terminal_games += 1.0f;
        env->log.terminal_rewards += actor_reward;
    }
    if (is_truncation) {
        env->log.truncations += 1.0f;
    }
}

static inline void cr_ocean_reset_state(ChainReactionOcean* env) {
    cr_init(&env->state);
    cr_set_active_region(&env->state, env->active_width, env->active_height);
    env->current_player = 1;
    cr_ocean_write_observation(env);
}

void c_reset(ChainReactionOcean* env) {
    cr_ocean_reset_state(env);
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
}

void c_step(ChainReactionOcean* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    int16_t wave_count = 0;
    int8_t wave_log_truncated = 0;
    int ok = cr_step_with_stats(&env->state, action, (int8_t)env->current_player, &wave_count, &wave_log_truncated);
    if (ok != 1) {
        // Illegal move: acting player forfeits (-1), opponent gains (+1).
        env->log.illegal_moves += 1.0f;
        env->rewards[0] = -1.0f;
        env->terminals[0] = 1.0f;
        float p1_result = env->current_player == 1 ? -1.0f : 1.0f;
        cr_ocean_add_game_log(env, p1_result, -1.0f, 1, 0);
        cr_ocean_reset_state(env);
        return;
    }

    env->log.cascade_depth += (float)wave_count;
    env->log.cascade_events += 1.0f;
    if ((float)wave_count > env->log.max_cascade_depth) {
        env->log.max_cascade_depth = (float)wave_count;
    }
    env->log.cascade_log_truncations += (float)wave_log_truncated;

    int8_t winner = cr_get_winner(&env->state);
    if (winner != 0) {
        float my_reward = winner == env->current_player ? 1.0f : -1.0f;
        env->rewards[0] = my_reward;
        env->terminals[0] = 1.0f;
        float p1_result = winner == 1 ? 1.0f : -1.0f;
        cr_ocean_add_game_log(env, p1_result, my_reward, 1, 0);
        cr_ocean_reset_state(env);
        return;
    }

    if (env->max_turns > 0 && env->state.turn_count >= env->max_turns) {
        env->terminals[0] = 1.0f;
        cr_ocean_add_game_log(env, 0.0f, 0.0f, 0, 1);
        cr_ocean_reset_state(env);
        return;
    }

    env->current_player = env->current_player == 1 ? 2 : 1;
    cr_ocean_write_observation(env);
}

void c_render(ChainReactionOcean* env) {
    (void)env;
}

void c_close(ChainReactionOcean* env) {
    (void)env;
}
