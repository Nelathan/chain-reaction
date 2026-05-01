import assert from "node:assert/strict";
import { ChainReaction } from "../core/game.ts";

const fixture = JSON.parse(ChainReaction.generateFixture());
const cases = new Map(fixture.cases.map((item) => [item.name, item]));

function getCase(name) {
  const item = cases.get(name);
  assert.ok(item, `missing fixture case: ${name}`);
  return item;
}

{
  const item = getCase("first-move-no-winner");
  assert.equal(item.winner, 0);
  assert.equal(item.lastMoveExploded, 0);
  assert.equal(item.playersAliveMask, 1);
}

{
  const item = getCase("opposing-pressure-cancels");
  assert.equal(item.lastMoveExploded, 1);
  assert.equal(item.tokens[1], 0);
  assert.equal(item.owners[1], 0);
  assert.equal(item.playersAliveMask, 3);
  assert.equal(item.winner, 0);
}

{
  const item = getCase("same-owner-pressure-stacks");
  assert.equal(item.lastMoveExploded, 1);
  assert.equal(item.tokens[1], 2);
  assert.equal(item.owners[1], 1);
}

{
  const item = getCase("source-keeps-residual-owner");
  assert.equal(item.tokens[1], 1);
  assert.equal(item.owners[1], 1);
}

{
  const item = getCase("source-clears-when-empty");
  assert.equal(item.tokens[1], 0);
  assert.equal(item.owners[1], 0);
}

{
  const item = getCase("elimination-after-explosion");
  assert.equal(item.lastMoveExploded, 1);
  assert.equal(item.playersAliveMask, 1);
  assert.equal(item.winner, 1);
}

{
  const item = getCase("invalid-move-does-not-mutate");
  assert.equal(item.invalidMoveAccepted, false);
  assert.equal(item.tokens[0], 1);
  assert.equal(item.owners[0], 1);
}

{
  const item = getCase("first-move-no-winner");
  assert.equal(item.legalP2[0], 0);
  assert.equal(item.observationP1[0], -1);
  assert.equal(item.observationP2[0], 1);
}

console.log("Core expected-semantics tests passed.");
