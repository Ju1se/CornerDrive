const assert = require("assert");
const { ethers } = require("hardhat");

async function deployAuditFixture() {
  const [owner, oracle, honestVehicle, rareVehicle, fraudVehicle] = await ethers.getSigners();
  const Audit = await ethers.getContractFactory("FLPGAudit");
  const audit = await Audit.deploy();
  await audit.waitForDeployment();
  await (await audit.setOracle(oracle.address)).wait();

  const stake = ethers.parseEther("1");
  await (await audit.connect(honestVehicle).registerVehicle({ value: stake })).wait();
  await (await audit.connect(rareVehicle).registerVehicle({ value: stake })).wait();
  await (await audit.connect(fraudVehicle).registerVehicle({ value: stake })).wait();

  return { audit, owner, oracle, honestVehicle, rareVehicle, fraudVehicle, stake };
}

async function expectRevert(txPromise, expectedMessage) {
  try {
    const tx = await txPromise;
    if (tx && typeof tx.wait === "function") {
      await tx.wait();
    }
    assert.fail(`Expected revert containing "${expectedMessage}"`);
  } catch (error) {
    assert(
      error.message.includes(expectedMessage),
      `Expected revert containing "${expectedMessage}", got "${error.message}"`
    );
  }
}

describe("FLPGAudit", function () {
  it("applies round-scoped economic multipliers in a single batch", async function () {
    const { audit, oracle, honestVehicle, rareVehicle, fraudVehicle, stake } = await deployAuditFixture();

    await (await audit.connect(oracle).setRoundEconomicPolicy(7, 12000, 15000, 12000)).wait();

    const settlementId = ethers.keccak256(ethers.toUtf8Bytes("round-7-batch-1"));
    await (
      await audit.connect(oracle).settleBatch(
        7,
        [honestVehicle.address],
        [rareVehicle.address],
        [fraudVehicle.address],
        settlementId
      )
    ).wait();

    for (let index = 2; index <= 5; index += 1) {
      const followUpSettlementId = ethers.keccak256(ethers.toUtf8Bytes(`round-7-batch-${index}`));
      await (
        await audit.connect(oracle).settleBatch(
          7,
          [honestVehicle.address],
          [],
          [],
          followUpSettlementId
        )
      ).wait();
    }

    const honestStats = await audit.getVehicleStats(honestVehicle.address);
    const rareStats = await audit.getVehicleStats(rareVehicle.address);
    const fraudStats = await audit.getVehicleStats(fraudVehicle.address);
    const roundPolicy = await audit.getRoundEconomicPolicy(7);

    assert.strictEqual(honestStats[0], 6n);
    assert.strictEqual(honestStats[1], 5n);

    assert.strictEqual(rareStats[0], 12n);
    assert.strictEqual(rareStats[1], 1n);
    assert.strictEqual(rareStats[3], 1n);

    assert.strictEqual(fraudStats[0], -75n);
    assert.strictEqual(fraudStats[2], 1n);
    assert.strictEqual(fraudStats[4], stake / 4n);

    assert.strictEqual(roundPolicy[0], 12000n);
    assert.strictEqual(roundPolicy[1], 15000n);
    assert.strictEqual(roundPolicy[2], 12000n);
    assert.strictEqual(roundPolicy[3], true);
    assert.strictEqual(roundPolicy[4], true);

    assert.strictEqual(await audit.processedSettlementIds(settlementId), true);
  });

  it("prevents replay and freezes round policy after settlement starts", async function () {
    const { audit, oracle, honestVehicle } = await deployAuditFixture();

    await (await audit.connect(oracle).setRoundEconomicPolicy(9, 10000, 10000, 10000)).wait();

    const settlementId = ethers.keccak256(ethers.toUtf8Bytes("round-9-batch-1"));
    await (
      await audit.connect(oracle).settleBatch(
        9,
        [honestVehicle.address],
        [],
        [],
        settlementId
      )
    ).wait();

    await expectRevert(
      audit.connect(oracle).settleBatch(9, [honestVehicle.address], [], [], settlementId),
      "Settlement already processed"
    );

    await expectRevert(
      audit.connect(oracle).setRoundEconomicPolicy(9, 12000, 12000, 11000),
      "Round locked"
    );
  });
});
