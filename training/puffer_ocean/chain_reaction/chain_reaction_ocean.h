#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"
typedef struct GameState GameState;
typedef struct WaveLog WaveLog;
#define inline static inline
#include "core/chain_reaction.hpp"
#undef inline

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
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
    GameState state;
} ChainReactionOcean;

static inline void cr_ocean_write_observation(ChainReactionOcean* env) {
    int8_t obs[CR_CELLS];
    cr_write_observation(&env->state, (int8_t)env->current_player, obs);
    for (int i = 0; i < CR_CELLS; ++i) {
        env->observations[i] = (float)obs[i];
    }
}

static inline void cr_ocean_add_log(ChainReactionOcean* env, float reward) {
    env->log.score += reward;
    env->log.episode_return += reward;
    env->log.episode_length += (float)env->state.turn_count;
    env->log.perf += reward > 0.0f ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

void c_reset(ChainReactionOcean* env) {
    cr_init(&env->state);
    env->current_player = 1;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    cr_ocean_write_observation(env);
}

void c_step(ChainReactionOcean* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    int ok = cr_step(&env->state, action, (int8_t)env->current_player);
    if (ok != 1) {
        env->rewards[0] = -1.0f;
        env->terminals[0] = 1.0f;
        cr_ocean_add_log(env, env->rewards[0]);
        c_reset(env);
        return;
    }

    int8_t winner = cr_get_winner(&env->state);
    if (winner != 0) {
        env->rewards[0] = winner == env->current_player ? 1.0f : -1.0f;
        env->terminals[0] = 1.0f;
        cr_ocean_add_log(env, env->rewards[0]);
        c_reset(env);
        return;
    }

    if (env->max_turns > 0 && env->state.turn_count >= env->max_turns) {
        env->terminals[0] = 1.0f;
        cr_ocean_add_log(env, 0.0f);
        c_reset(env);
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
