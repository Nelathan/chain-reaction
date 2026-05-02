#include "chain_reaction_ocean.h"

int main(void) {
    ChainReactionOcean env;
    memset(&env, 0, sizeof(env));
    float observations[CR_CELLS];
    float actions[1];
    float rewards[1];
    float terminals[1];
    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.num_agents = 1;
    env.max_turns = 4096;
    c_reset(&env);
    return 0;
}
