#include "chain_reaction_ocean.h"

#define OBS_SIZE CR_CELLS
#define NUM_ATNS 1
#define ACT_SIZES {CR_CELLS}
#define OBS_TENSOR_T FloatTensor

#define Env ChainReactionOcean
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->max_turns = (int)dict_get(kwargs, "max_turns")->value;
    env->active_width = (int)dict_get(kwargs, "active_width")->value;
    env->active_height = (int)dict_get(kwargs, "active_height")->value;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
    dict_set(out, "illegal_moves", log->illegal_moves);
    dict_set(out, "truncations", log->truncations);
    dict_set(out, "truncation_rate", log->truncations);
    dict_set(out, "terminal_games", log->terminal_games);
    dict_set(out, "terminal_rate", log->terminal_games);
    dict_set(out, "mean_terminal_actor_reward", log->terminal_games > 0.0f ? log->terminal_rewards / log->terminal_games : 0.0f);
    dict_set(out, "cascade_depth", log->n > 0.0f ? log->cascade_depth / log->n : 0.0f);
    dict_set(out, "mean_cascade_depth", log->cascade_events > 0.0f ? log->cascade_depth / log->cascade_events : 0.0f);
    dict_set(out, "max_cascade_depth", log->max_cascade_depth);
    dict_set(out, "cascade_log_truncations", log->cascade_log_truncations);
    dict_set(out, "winrate", log->p1_games > 0.0f ? log->p1_wins / log->p1_games : 0.0f);
    dict_set(out, "terminal_winrate", log->terminal_games > 0.0f ? log->p1_wins / log->terminal_games : 0.0f);
}
