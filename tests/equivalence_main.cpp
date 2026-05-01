#include <stdio.h>

#include "../core/chain_reaction.hpp"

static void load_fixture(GameState* game, const int* tokens, const int* owners, int count, int alive) {
    cr_init(game);
    for (int i = 0; i < count; ++i) {
        game->tokens[i] = (int8_t)tokens[i];
        game->owners[i] = (int8_t)owners[i];
    }
    game->players_alive_mask = (int8_t)alive;
}

static void print_array(const int8_t* values, int count) {
    for (int i = 0; i < count; ++i) {
        if (i > 0) printf(",");
        printf("%d", (int)values[i]);
    }
}

static void print_case(const char* name, const GameState* game, const WaveLog* log, int include_invalid, int invalid_accepted) {
    int8_t legal_p1[CR_CELLS];
    int8_t legal_p2[CR_CELLS];
    int8_t observation_p1[CR_CELLS];
    int8_t observation_p2[CR_CELLS];

    cr_write_legal_actions(game, (int8_t)1, legal_p1);
    cr_write_legal_actions(game, (int8_t)2, legal_p2);
    cr_write_observation(game, (int8_t)1, observation_p1);
    cr_write_observation(game, (int8_t)2, observation_p2);

    printf("{\"name\":\"%s\",\"tokens\":[", name);
    print_array(game->tokens, CR_CELLS);

    printf("],\"owners\":[");
    print_array(game->owners, CR_CELLS);

    printf("],\"playersAliveMask\":%d", (int)game->players_alive_mask);
    printf(",\"lastMoveExploded\":%d", (int)game->last_move_exploded);
    printf(",\"waveCount\":%d", (int)log->wave_count);
    printf(",\"waveLogTruncated\":%d", (int)log->wave_log_truncated);
    printf(",\"waveExploded\":[");
    print_array(log->wave_exploded, CR_MAX_LOGGED_WAVES * CR_CELLS);
    printf("],\"waveTokens\":[");
    print_array(log->wave_tokens, CR_MAX_LOGGED_WAVES * CR_CELLS);
    printf("],\"waveOwners\":[");
    print_array(log->wave_owners, CR_MAX_LOGGED_WAVES * CR_CELLS);
    printf("],\"winner\":%d", (int)cr_get_winner(game));
    printf(",\"legalP1\":[");
    print_array(legal_p1, CR_CELLS);
    printf("],\"legalP2\":[");
    print_array(legal_p2, CR_CELLS);
    printf("],\"observationP1\":[");
    print_array(observation_p1, CR_CELLS);
    printf("],\"observationP2\":[");
    print_array(observation_p2, CR_CELLS);
    printf("]");
    if (include_invalid) {
        printf(",\"invalidMoveAccepted\":%s", invalid_accepted ? "true" : "false");
    }
    printf("}");
}

static void play_moves(GameState* game, WaveLog* log, const int moves[][2], int count) {
    cr_init(game);
    cr_init_wave_log(log);
    for (int i = 0; i < count; ++i) {
        cr_step_with_log(game, log, moves[i][0], (int8_t)moves[i][1]);
    }
}

int main(void) {
    GameState game;
    WaveLog log;

    printf("{\"cases\":[");

    const int scripted[][2] = {
        {0, 1},
        {1, 2},
        {0, 1},
        {1, 2},
        {9, 1},
        {8, 2},
        {9, 1},
        {8, 2},
        {0, 1},
    };
    play_moves(&game, &log, scripted, 9);
    print_case("scripted-cascade", &game, &log, 0, 0);

    const int first_move[][2] = {{0, 1}};
    play_moves(&game, &log, first_move, 1);
    printf(",");
    print_case("first-move-no-winner", &game, &log, 0, 0);

    const int opposing_tokens[] = {2, 0, 3};
    const int opposing_owners[] = {1, 0, 2};
    load_fixture(&game, opposing_tokens, opposing_owners, 3, 3);
    cr_step_with_log(&game, &log, 9, (int8_t)1);
    printf(",");
    print_case("opposing-pressure-cancels", &game, &log, 0, 0);

    const int same_tokens[] = {2, 0, 3};
    const int same_owners[] = {1, 0, 1};
    load_fixture(&game, same_tokens, same_owners, 3, 3);
    cr_step_with_log(&game, &log, 9, (int8_t)1);
    printf(",");
    print_case("same-owner-pressure-stacks", &game, &log, 0, 0);

    const int wait_tokens[] = {1, 0, 1, 0, 0, 0, 0, 0, 1};
    const int wait_owners[] = {1, 0, 1, 0, 0, 0, 0, 0, 1};
    load_fixture(&game, wait_tokens, wait_owners, 9, 3);
    cr_step_with_log(&game, &log, 1, (int8_t)1);
    printf(",");
    print_case("incoming-critical-waits-for-next-wave", &game, &log, 0, 0);

    const int residual_tokens[] = {0, 3};
    const int residual_owners[] = {0, 1};
    load_fixture(&game, residual_tokens, residual_owners, 2, 3);
    cr_step_with_log(&game, &log, 1, (int8_t)1);
    printf(",");
    print_case("source-keeps-residual-owner", &game, &log, 0, 0);

    const int clear_tokens[] = {0, 2};
    const int clear_owners[] = {0, 1};
    load_fixture(&game, clear_tokens, clear_owners, 2, 3);
    cr_step_with_log(&game, &log, 1, (int8_t)1);
    printf(",");
    print_case("source-clears-when-empty", &game, &log, 0, 0);

    const int eliminate_tokens[] = {1, 1};
    const int eliminate_owners[] = {1, 2};
    load_fixture(&game, eliminate_tokens, eliminate_owners, 2, 3);
    cr_step_with_log(&game, &log, 0, (int8_t)1);
    printf(",");
    print_case("elimination-after-explosion", &game, &log, 0, 0);

    const int corner_tokens[] = {1};
    const int corner_owners[] = {1};
    load_fixture(&game, corner_tokens, corner_owners, 1, 3);
    cr_step_with_log(&game, &log, 0, (int8_t)1);
    printf(",");
    print_case("critical-mass-corner", &game, &log, 0, 0);

    const int edge_tokens[] = {0, 2};
    const int edge_owners[] = {0, 1};
    load_fixture(&game, edge_tokens, edge_owners, 2, 3);
    cr_step_with_log(&game, &log, 1, (int8_t)1);
    printf(",");
    print_case("critical-mass-edge", &game, &log, 0, 0);

    const int center_tokens[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 3};
    const int center_owners[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    load_fixture(&game, center_tokens, center_owners, 10, 3);
    cr_step_with_log(&game, &log, 9, (int8_t)1);
    printf(",");
    print_case("critical-mass-center", &game, &log, 0, 0);

    cr_init(&game);
    cr_init_wave_log(&log);
    cr_step_with_log(&game, &log, 0, (int8_t)1);
    int invalid = cr_step_with_log(&game, &log, 0, (int8_t)2);
    printf(",");
    print_case("invalid-move-does-not-mutate", &game, &log, 1, invalid);

    printf("]}");
    return 0;
}
