export const WIDTH = 8;
export const HEIGHT = 8;
export const CELLS = WIDTH * HEIGHT;
export const PLAYER_COUNT = 2;

type Move = [number, number];

export type FixtureCase = {
  name: string;
  tokens: number[];
  owners: number[];
  playersSeenMask: number;
  playersAliveMask: number;
  lastMoveExploded: number;
  winner: number;
  legalP1: number[];
  legalP2: number[];
  observationP1: number[];
  observationP2: number[];
  invalidMoveAccepted?: boolean;
};

export class ChainReaction {
  tokens: Int8Array;
  owners: Int8Array;
  private nextTokens: Int8Array;
  private nextOwners: Int8Array;
  private pressure: Int8Array;
  turnCount: number;
  playersSeenMask: number;
  playersAliveMask: number;
  lastMoveExploded: number;
  winner: number;

  constructor() {
    this.tokens = new Int8Array(CELLS);
    this.owners = new Int8Array(CELLS);
    this.nextTokens = new Int8Array(CELLS);
    this.nextOwners = new Int8Array(CELLS);
    this.pressure = new Int8Array(PLAYER_COUNT * CELLS);
    this.turnCount = 0;
    this.playersSeenMask = 0;
    this.playersAliveMask = 0;
    this.lastMoveExploded = 0;
    this.winner = 0;
  }

  reset(): void {
    this.tokens.fill(0);
    this.owners.fill(0);
    this.nextTokens.fill(0);
    this.nextOwners.fill(0);
    this.pressure.fill(0);
    this.turnCount = 0;
    this.playersSeenMask = 0;
    this.playersAliveMask = 0;
    this.lastMoveExploded = 0;
    this.winner = 0;
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
    if (playerId < 1 || playerId > PLAYER_COUNT) return false;
    if (actionIdx < 0 || actionIdx >= CELLS) return false;
    if (this.winner !== 0) return false;
    if ((this.playersSeenMask & this.playerBit(playerId)) !== 0 && (this.playersAliveMask & this.playerBit(playerId)) === 0) {
      return false;
    }
    return this.owners[actionIdx] === 0 || this.owners[actionIdx] === playerId;
  }

  step(actionIdx: number, playerId: number): boolean {
    if (!this.isLegalMove(actionIdx, playerId)) return false;

    this.tokens[actionIdx]++;
    this.owners[actionIdx] = playerId;
    this.turnCount++;
    this.playersSeenMask |= this.playerBit(playerId);
    this.lastMoveExploded = 0;
    this.winner = 0;

    let unstable = true;
    while (unstable) {
      unstable = this.resolveWave();
      if (unstable) this.lastMoveExploded = 1;
    }

    this.updateAliveAndWinner();
    return true;
  }

  getWinner(): number {
    return this.winner;
  }

  writeLegalActions(playerId: number, out: Int8Array): void {
    for (let i = 0; i < CELLS; i++) {
      out[i] = this.isLegalMove(i, playerId) ? 1 : 0;
    }
  }

  writeObservation(playerId: number, out: Int8Array): void {
    for (let i = 0; i < CELLS; i++) {
      const owner = this.owners[i];
      if (owner === 0) {
        out[i] = 0;
      } else if (owner === playerId) {
        out[i] = this.tokens[i] - this.getMass(i);
      } else {
        out[i] = (this.tokens[i] - this.getMass(i)) * -1;
      }
    }
  }

  static generateFixture(): string {
    const cases: FixtureCase[] = [
      this.caseFromMoves("scripted-cascade", [
        [0, 1],
        [1, 2],
        [0, 1],
        [1, 2],
        [9, 1],
        [8, 2],
        [9, 1],
        [8, 2],
        [0, 1],
      ]),
      this.caseFromMoves("first-move-no-winner", [[0, 1]]),
      this.opposingPressureCancels(),
      this.sameOwnerPressureStacks(),
      this.incomingCriticalWaitsForNextWave(),
      this.sourceKeepsResidualOwner(),
      this.sourceClearsWhenEmpty(),
      this.eliminationAfterExplosion(),
      this.invalidMoveDoesNotMutate(),
    ];

    return JSON.stringify({ cases });
  }

  private resolveWave(): boolean {
    let exploded = false;

    this.nextTokens.set(this.tokens);
    this.nextOwners.set(this.owners);
    this.pressure.fill(0);

    for (let i = 0; i < CELLS; i++) {
      const owner = this.owners[i];
      if (owner === 0) continue;

      const mass = this.getMass(i);
      if (this.tokens[i] >= mass) {
        exploded = true;
        this.nextTokens[i] -= mass;
        if (this.nextTokens[i] === 0) this.nextOwners[i] = 0;

        const x = i % WIDTH;
        const y = Math.floor(i / WIDTH);

        if (y > 0) this.addPressure(i - WIDTH, owner);
        if (y < HEIGHT - 1) this.addPressure(i + WIDTH, owner);
        if (x > 0) this.addPressure(i - 1, owner);
        if (x < WIDTH - 1) this.addPressure(i + 1, owner);
      }
    }

    if (!exploded) return false;

    this.applyPressure();
    this.tokens.set(this.nextTokens);
    this.owners.set(this.nextOwners);
    return true;
  }

  private addPressure(idx: number, owner: number): void {
    this.pressure[(owner - 1) * CELLS + idx]++;
  }

  private applyPressure(): void {
    for (let i = 0; i < CELLS; i++) {
      const p1 = this.pressure[i];
      const p2 = this.pressure[CELLS + i];
      const net = p1 - p2;

      if (net > 0) {
        this.nextTokens[i] += net;
        this.nextOwners[i] = 1;
      } else if (net < 0) {
        this.nextTokens[i] -= net;
        this.nextOwners[i] = 2;
      } else if (this.nextTokens[i] === 0) {
        this.nextOwners[i] = 0;
      }
    }
  }

  private updateAliveAndWinner(): void {
    let alive = 0;
    for (let i = 0; i < CELLS; i++) {
      const owner = this.owners[i];
      if (owner >= 1 && owner <= PLAYER_COUNT) alive |= this.playerBit(owner);
    }

    this.playersAliveMask = alive;

    if (this.lastMoveExploded === 0) return;
    if (this.countBits(this.playersSeenMask) < 2) return;
    if (this.countBits(alive) === 1) this.winner = this.maskToPlayer(alive);
  }

  private playerBit(playerId: number): number {
    return 1 << (playerId - 1);
  }

  private countBits(mask: number): number {
    let count = 0;
    let value = mask;
    while (value !== 0) {
      count += value & 1;
      value >>= 1;
    }
    return count;
  }

  private maskToPlayer(mask: number): number {
    for (let player = 1; player <= PLAYER_COUNT; player++) {
      if (mask === this.playerBit(player)) return player;
    }
    return 0;
  }

  private static caseFromMoves(name: string, moves: Move[]): FixtureCase {
    const game = new ChainReaction();
    for (const [idx, player] of moves) game.step(idx, player);
    return this.fixtureCase(name, game);
  }

  private static opposingPressureCancels(): FixtureCase {
    const game = new ChainReaction();
    game.loadForFixture([2, 0, 3], [1, 0, 2], 0b11, 0b11);
    game.step(9, 1);
    return this.fixtureCase("opposing-pressure-cancels", game);
  }

  private static sameOwnerPressureStacks(): FixtureCase {
    const game = new ChainReaction();
    game.loadForFixture([2, 0, 3], [1, 0, 1], 0b1, 0b1);
    game.step(9, 1);
    return this.fixtureCase("same-owner-pressure-stacks", game);
  }

  private static incomingCriticalWaitsForNextWave(): FixtureCase {
    const game = new ChainReaction();
    game.loadForFixture([1, 0, 1, 0, 0, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0, 0, 0, 1], 0b1, 0b1);
    game.step(1, 1);
    return this.fixtureCase("incoming-critical-waits-for-next-wave", game);
  }

  private static sourceKeepsResidualOwner(): FixtureCase {
    const game = new ChainReaction();
    game.loadForFixture([0, 3], [0, 1], 0b1, 0b1);
    game.step(1, 1);
    return this.fixtureCase("source-keeps-residual-owner", game);
  }

  private static sourceClearsWhenEmpty(): FixtureCase {
    const game = new ChainReaction();
    game.loadForFixture([0, 2], [0, 1], 0b1, 0b1);
    game.step(1, 1);
    return this.fixtureCase("source-clears-when-empty", game);
  }

  private static eliminationAfterExplosion(): FixtureCase {
    const game = new ChainReaction();
    game.loadForFixture([1, 1], [1, 2], 0b11, 0b11);
    game.step(0, 1);
    return this.fixtureCase("elimination-after-explosion", game);
  }

  private static invalidMoveDoesNotMutate(): FixtureCase {
    const game = new ChainReaction();
    game.step(0, 1);
    const accepted = game.step(0, 2);
    return { ...this.fixtureCase("invalid-move-does-not-mutate", game), invalidMoveAccepted: accepted };
  }

  private static fixtureCase(name: string, game: ChainReaction): FixtureCase {
    const legalP1 = new Int8Array(CELLS);
    const legalP2 = new Int8Array(CELLS);
    const observationP1 = new Int8Array(CELLS);
    const observationP2 = new Int8Array(CELLS);
    game.writeLegalActions(1, legalP1);
    game.writeLegalActions(2, legalP2);
    game.writeObservation(1, observationP1);
    game.writeObservation(2, observationP2);

    return {
      name,
      tokens: Array.from(game.tokens),
      owners: Array.from(game.owners),
      playersSeenMask: game.playersSeenMask,
      playersAliveMask: game.playersAliveMask,
      lastMoveExploded: game.lastMoveExploded,
      winner: game.getWinner(),
      legalP1: Array.from(legalP1),
      legalP2: Array.from(legalP2),
      observationP1: Array.from(observationP1),
      observationP2: Array.from(observationP2),
    };
  }

  private loadForFixture(tokens: number[], owners: number[], seen: number, alive: number): void {
    this.reset();
    for (let i = 0; i < tokens.length; i++) {
      this.tokens[i] = tokens[i];
      this.owners[i] = owners[i];
    }
    this.playersSeenMask = seen;
    this.playersAliveMask = alive;
  }
}

declare const process: { argv: string[]; stdout: { write(value: string): void } } | undefined;

if (typeof process !== "undefined" && process.argv[1]?.endsWith("game.ts")) {
  process.stdout.write(ChainReaction.generateFixture());
}
