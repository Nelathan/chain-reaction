#include <godot_cpp/classes/node2d.hpp>
#include "../../core-logic/chain_reaction.hpp"

namespace godot {
    class ChainReactionNode : public Node2D {
        GDCLASS(ChainReactionNode, Node2D)
    private:
        GameState game_state;
    public:
        void _process(double delta) override {
            // Read game_state.cells[i] and update Sprites
            // Render logic only.
        }
    };
}
