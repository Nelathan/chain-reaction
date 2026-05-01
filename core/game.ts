export const WIDTH = 8;
export const HEIGHT = 8;
export const CELLS = WIDTH * HEIGHT;

export type Fixture = {
  tokens: number[];
  owners: number[];
  winner: number;
  firstMoveWinner: number;
  invalidMoveAccepted: boolean;
};

export class ChainReaction {
  tokens: Int8Array;
  owners: Int8Array;
  private nextTokens: Int8Array;
  private nextOwners: Int8Array;
  private turnCount: number;

  constructor() {
    this.tokens = new Int8Array(CELLS);
    this.owners = new Int8Array(CELLS);
    this.nextTokens = new Int8Array(CELLS);
    this.nextOwners = new Int8Array(CELLS);
    this.turnCount = 0;
  }

  reset(): void {
    this.tokens.fill(0);
    this.owners.fill(0);
    this.nextTokens.fill(0);
    this.nextOwners.fill(0);
    this.turnCount = 0;
  }

  getMass(idx: number): number {
    const x = idx % WIDTH;
    const y = Math.floor(idx / WIDTH);
    let mass = 4;
    if (x === 0 || x === WIDTH - 1) mass--;
    if (y === 0 || y === HEIGHT - 1) mass--;
    return mass;
  }

  isLegalMove(actionIdx: number, playerId: number): boolean {
    if (playerId !== 1 && playerId !== 2) return false;
    if (actionIdx < 0 || actionIdx >= CELLS) return false;
    return this.owners[actionIdx] === 0 || this.owners[actionIdx] === playerId;
  }

  step(actionIdx: number, playerId: number): boolean {
    if (!this.isLegalMove(actionIdx, playerId)) return false;

    this.tokens[actionIdx]++;
    this.owners[actionIdx] = playerId;
    this.turnCount++;

    let unstable = true;
    while (unstable) {
      unstable = false;

      this.nextTokens.set(this.tokens);
      this.nextOwners.set(this.owners);

      for (let i = 0; i < CELLS; i++) {
        const mass = this.getMass(i);
        if (this.tokens[i] >= mass) {
          unstable = true;
          const owner = this.owners[i];

          this.nextTokens[i] -= mass;
          if (this.nextTokens[i] === 0) this.nextOwners[i] = 0;

          const x = i % WIDTH;
          const y = Math.floor(i / WIDTH);

          // Opposing simultaneous scatters resolve deterministically by scan order.
          if (y > 0) this.scatter(i - WIDTH, owner);
          if (y < HEIGHT - 1) this.scatter(i + WIDTH, owner);
          if (x > 0) this.scatter(i - 1, owner);
          if (x < WIDTH - 1) this.scatter(i + 1, owner);
        }
      }

      this.tokens.set(this.nextTokens);
      this.owners.set(this.nextOwners);
    }

    return true;
  }

  getWinner(): number {
    if (this.turnCount < 2) return 0;

    let p1 = false;
    let p2 = false;

    for (let i = 0; i < CELLS; i++) {
      if (this.owners[i] === 1) p1 = true;
      if (this.owners[i] === 2) p2 = true;
      if (p1 && p2) return 0;
    }

    if (p1) return 1;
    if (p2) return 2;
    return 0;
  }

  static generateFixture(): string {
    const opening = new ChainReaction();
    opening.step(0, 1);

    const game = new ChainReaction();
    const moves: Array<[number, number]> = [
      [0, 1],
      [1, 2],
      [0, 1],
      [1, 2],
      [9, 1],
      [8, 2],
      [9, 1],
      [8, 2],
      [0, 1],
    ];

    for (const [idx, player] of moves) {
      game.step(idx, player);
    }

    const fixture: Fixture = {
      tokens: Array.from(game.tokens),
      owners: Array.from(game.owners),
      winner: game.getWinner(),
      firstMoveWinner: opening.getWinner(),
      invalidMoveAccepted: game.step(0, 2),
    };

    return JSON.stringify(fixture);
  }

  private scatter(idx: number, owner: number): void {
    this.nextTokens[idx]++;
    this.nextOwners[idx] = owner;
  }
}

declare const process: { argv: string[]; stdout: { write(value: string): void } } | undefined;

if (typeof process !== "undefined" && process.argv[1]?.endsWith("game.ts")) {
  process.stdout.write(ChainReaction.generateFixture());
}
