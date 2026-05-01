const WIDTH = 8;
const HEIGHT = 8;

type Board = {
  tokens: Int32Array; // Flat array: y * WIDTH + x
  owners: Int32Array; // 0=None, 1=P1, 2=P2
};

export class ChainReaction {
  current: Board;
  next: Board;

  constructor() {
    this.current = { tokens: new Int32Array(WIDTH * HEIGHT), owners: new Int32Array(WIDTH * HEIGHT) };
    this.next = { tokens: new Int32Array(WIDTH * HEIGHT), owners: new Int32Array(WIDTH * HEIGHT) };
  }

  getCriticalMass(x: number, y: number): number {
    let mass = 4;
    if (x === 0 || x === WIDTH - 1) mass--;
    if (y === 0 || y === HEIGHT - 1) mass--;
    return mass;
  }

  // Reference logic for double-buffered simultaneous explosions
  // (Full step logic mapped out previously)
}
