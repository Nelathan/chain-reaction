#include <stdio.h>

#include "../core/chain_reaction.hpp"

int main(void) {
    GameState opening;
    cr_init(&opening);
    cr_step(&opening, 0, (int8_t)1);

    GameState game;
    cr_init(&game);

    const int moves[][2] = {
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

    for (int i = 0; i < 9; ++i) {
        cr_step(&game, moves[i][0], (int8_t)moves[i][1]);
    }

    printf("{\"tokens\":[");
    for (int i = 0; i < CR_CELLS; ++i) {
        if (i > 0) printf(",");
        printf("%d", (int)game.tokens[i]);
    }

    printf("],\"owners\":[");
    for (int i = 0; i < CR_CELLS; ++i) {
        if (i > 0) printf(",");
        printf("%d", (int)game.owners[i]);
    }

    int invalid_move_accepted = cr_step(&game, 0, (int8_t)2);

    printf("],\"winner\":%d", (int)cr_get_winner(&game));
    printf(",\"firstMoveWinner\":%d", (int)cr_get_winner(&opening));
    printf(",\"invalidMoveAccepted\":%s}", invalid_move_accepted ? "true" : "false");
    return 0;
}
