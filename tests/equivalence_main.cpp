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

static void print_case(const char* name, const GameState* game, int include_invalid, int invalid_accepted) {
    int8_t legal_p1[CR_CELLS];
    int8_t legal_p2[CR_CELLS];
    int8_t observation_p1[CR_CELLS];
    int8_t observation_p2[CR_CELLS];

    cr_write_legal_actions(game, (int8_t)1, legal_p1);
    cr_write_legal_actions(game, (int8_t)2, legal_p2);
    cr_write_observation(game, (int8_t)1, observation_p1);
    cr_write_observation(game, (int8_t)2, observation_p2);

    printf("{\"name\":\"%s\",\"tokens\":[", name);
    for (int i = 0; i < CR_CELLS; ++i) {
        if (i > 0) printf(",");
        printf("%d", (int)game->tokens[i]);
    }

    printf("],\"owners\":[");
    for (int i = 0; i < CR_CELLS; ++i) {
        if (i > 0) printf(",");
        printf("%d", (int)game->owners[i]);
    }

    printf("],\"playersAliveMask\":%d", (int)game->players_alive_mask);
    printf(",\"lastMoveExploded\":%d", (int)game->last_move_exploded);
    printf(",\"winner\":%d", (int)cr_get_winner(game));
    printf(",\"legalP1\":[");
    for (int i = 0; i < CR_CELLS; ++i) {
        if (i > 0) printf(",");
        printf("%d", (int)legal_p1[i]);
    }
    printf("],\"legalP2\":[");
    for (int i = 0; i < CR_CELLS; ++i) {
        if (i > 0) printf(",");
        printf("%d", (int)legal_p2[i]);
    }
    printf("],\"observationP1\":[");
    for (int i = 0; i < CR_CELLS; ++i) {
        if (i > 0) printf(",");
        printf("%d", (int)observation_p1[i]);
    }
    printf("],\"observationP2\":[");
    for (int i = 0; i < CR_CELLS; ++i) {
        if (i > 0) printf(",");
        printf("%d", (int)observation_p2[i]);
    }
    printf("]");
    if (include_invalid) {
        printf(",\"invalidMoveAccepted\":%s", invalid_accepted ? "true" : "false");
    }
    printf("}");
}

static void play_moves(GameState* game, const int moves[][2], int count) {
    cr_init(game);
    for (int i = 0; i < count; ++i) {
        cr_step(game, moves[i][0], (int8_t)moves[i][1]);
    }
}

int main(void) {
    GameState game;

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
    play_moves(&game, scripted, 9);
    print_case("scripted-cascade", &game, 0, 0);

    const int first_move[][2] = {{0, 1}};
    play_moves(&game, first_move, 1);
    printf(",");
    print_case("first-move-no-winner", &game, 0, 0);

    const int opposing_tokens[] = {2, 0, 3};
    const int opposing_owners[] = {1, 0, 2};
    load_fixture(&game, opposing_tokens, opposing_owners, 3, 3);
    cr_step(&game, 9, (int8_t)1);
    printf(",");
    print_case("opposing-pressure-cancels", &game, 0, 0);

    const int same_tokens[] = {2, 0, 3};
    const int same_owners[] = {1, 0, 1};
    load_fixture(&game, same_tokens, same_owners, 3, 3);
    cr_step(&game, 9, (int8_t)1);
    printf(",");
    print_case("same-owner-pressure-stacks", &game, 0, 0);

    const int wait_tokens[] = {1, 0, 1, 0, 0, 0, 0, 0, 1};
    const int wait_owners[] = {1, 0, 1, 0, 0, 0, 0, 0, 1};
    load_fixture(&game, wait_tokens, wait_owners, 9, 3);
    cr_step(&game, 1, (int8_t)1);
    printf(",");
    print_case("incoming-critical-waits-for-next-wave", &game, 0, 0);

    const int residual_tokens[] = {0, 3};
    const int residual_owners[] = {0, 1};
    load_fixture(&game, residual_tokens, residual_owners, 2, 3);
    cr_step(&game, 1, (int8_t)1);
    printf(",");
    print_case("source-keeps-residual-owner", &game, 0, 0);

    const int clear_tokens[] = {0, 2};
    const int clear_owners[] = {0, 1};
    load_fixture(&game, clear_tokens, clear_owners, 2, 3);
    cr_step(&game, 1, (int8_t)1);
    printf(",");
    print_case("source-clears-when-empty", &game, 0, 0);

    const int eliminate_tokens[] = {1, 1};
    const int eliminate_owners[] = {1, 2};
    load_fixture(&game, eliminate_tokens, eliminate_owners, 2, 3);
    cr_step(&game, 0, (int8_t)1);
    printf(",");
    print_case("elimination-after-explosion", &game, 0, 0);

    cr_init(&game);
    cr_step(&game, 0, (int8_t)1);
    int invalid = cr_step(&game, 0, (int8_t)2);
    printf(",");
    print_case("invalid-move-does-not-mutate", &game, 1, invalid);

    printf("]}");
    return 0;
}
