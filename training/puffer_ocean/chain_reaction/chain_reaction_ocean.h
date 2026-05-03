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

static inline void cr_ocean_reset_state(ChainReactionOcean* env) {
    cr_init(&env->state);
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
    int ok = cr_step(&env->state, action, (int8_t)env->current_player);
    if (ok != 1) {
        // Illegal move: acting player forfeits (-1), opponent gains (+1).
        env->log.illegal_moves += 1.0f;
        env->rewards[0] = -1.0f;
        env->terminals[0] = 1.0f;
        cr_ocean_add_log(env, -1.0f);
        cr_ocean_add_log(env, 1.0f);
        cr_ocean_reset_state(env);
        return;
    }

    int8_t winner = cr_get_winner(&env->state);
    if (winner != 0) {
        // Zero-sum: log both winner (+1) and loser (-1) as separate entries.
        // In self-play the same policy plays both sides, so the losing player's
        // final turn (reward 0, game ongoing) is followed by the winner's
        // terminal move (+1).  Without the loser entry, the losing side never
        // sees an explicit -1 reward.  Two entries per game means
        // episode_return mean = 0 and perf = 0.5 for balanced play; track
        // winrate from player 1's perspective via p1_wins / p1_games.
        float my_reward = winner == env->current_player ? 1.0f : -1.0f;
        env->rewards[0] = my_reward;
        env->terminals[0] = 1.0f;
        cr_ocean_add_log(env, my_reward);
        cr_ocean_add_log(env, -my_reward);
        env->log.p1_games += 1.0f;
        if (winner == 1) env->log.p1_wins += 1.0f;
        cr_ocean_reset_state(env);
        return;
    }

    if (env->max_turns > 0 && env->state.turn_count >= env->max_turns) {
        env->terminals[0] = 1.0f;
        cr_ocean_add_log(env, 0.0f);
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
