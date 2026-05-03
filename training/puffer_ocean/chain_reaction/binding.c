#include "chain_reaction_ocean.h"

#define OBS_SIZE 64
#define NUM_ATNS 1
#define ACT_SIZES {64}
#define OBS_TENSOR_T FloatTensor

#define Env ChainReactionOcean
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->max_turns = (int)dict_get(kwargs, "max_turns")->value;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
    dict_set(out, "illegal_moves", log->illegal_moves);
    dict_set(out, "winrate", log->p1_games > 0.0f ? log->p1_wins / log->p1_games : 0.0f);
}
