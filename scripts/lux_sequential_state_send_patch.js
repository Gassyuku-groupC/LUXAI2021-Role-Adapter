"use strict";

// Dimensions queues every state line for both agents through Promise.all.
// Dense late-game boards can create thousands of concurrent writes on Windows.
// Bound concurrent pipe writes so dense boards cannot flood child-process stdin.
// The patch is process-local: it does not modify node_modules or agent packages.
const fs = require("fs");
const logicPath = "@lux-ai/2021-challenge/lib/es5/logic";
const logicFile = require.resolve(logicPath);
const originalJsLoader = require.extensions[".js"];

require.extensions[".js"] = function loadWithTerminalFix(module, filename) {
  if (filename !== logicFile) {
    return originalJsLoader(module, filename);
  }

  let source = fs.readFileSync(filename, "utf8");
  const marker = "                        game.runCooldowns();\r\n                        /** Agent Update Section */";
  const fallbackMarker = marker.replace("\r\n", "\n");
  const selectedMarker = source.includes(marker) ? marker : fallbackMarker;
  if (!source.includes(selectedMarker)) {
    throw new Error(`Lux terminal patch marker not found in ${filename}`);
  }
  const newline = selectedMarker.includes("\r\n") ? "\r\n" : "\n";
  const terminalBranch = [
    "                        game.runCooldowns();",
    "                        // Avoid sending an unusable final board/D_DONE through blocked agent pipes.",
    "                        if (matchOver) {",
    "                            if (game.replay) {",
    "                                game.replay.writeOut(this.getResults(match));",
    "                            }",
    "                            return [2 /*return*/, 'finished'];",
    "                        }",
    "                        /** Agent Update Section */",
  ].join(newline);
  source = source.replace(selectedMarker, terminalBranch);
  module._compile(source, filename);
};

const { LuxDesignLogic } = require(logicPath);
require.extensions[".js"] = originalJsLoader;

LuxDesignLogic.sendAllAgentsGameInformation = async function sendSequential(match) {
  const game = match.state.game;
  const messages = [];
  const teams = [0, 1];
  const batchSize = Number.parseInt(process.env.LUX_STATE_SEND_BATCH_SIZE || "64", 10);

  for (const team of teams) {
    messages.push(`rp ${team} ${game.state.teamStates[team].researchPoints}`);
  }
  for (const cell of game.map.resources) {
    messages.push(
      `r ${cell.resource.type} ${cell.pos.x} ${cell.pos.y} ${cell.resource.amount}`
    );
  }
  for (const team of teams) {
    for (const unit of game.getTeamsUnits(team).values()) {
      messages.push(
        `u ${unit.type} ${team} ${unit.id} ${unit.pos.x} ${unit.pos.y} ` +
        `${unit.cooldown} ${unit.cargo.wood} ${unit.cargo.coal} ${unit.cargo.uranium}`
      );
    }
  }
  for (const city of game.cities.values()) {
    messages.push(`c ${city.team} ${city.id} ${city.fuel} ${city.getLightUpkeep()}`);
  }
  for (const city of game.cities.values()) {
    for (const cell of city.citycells) {
      messages.push(
        `ct ${city.team} ${city.id} ${cell.pos.x} ${cell.pos.y} ${cell.citytile.cooldown}`
      );
    }
  }
  for (let y = 0; y < game.map.height; y += 1) {
    for (let x = 0; x < game.map.width; x += 1) {
      const road = game.map.getCell(x, y).getRoad();
      if (road !== 0) messages.push(`ccd ${x} ${y} ${road}`);
    }
  }

  let pending = [];
  for (const message of messages) {
    for (const agent of match.agents) {
      if (!agent.isTerminated()) pending.push(match.send(message, agent));
    }
    if (pending.length >= batchSize) {
      await Promise.all(pending);
      pending = [];
    }
  }
  if (pending.length > 0) await Promise.all(pending);
};
