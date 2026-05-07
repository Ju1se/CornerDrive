const { ethers } = require("hardhat");

async function main() {
  console.log("🚀 Deploying FLPG Contracts...\n");

  // Get deployer
  const [deployer] = await ethers.getSigners();
  console.log("Deployer address:", deployer.address);

  const balance = await ethers.provider.getBalance(deployer.address);
  console.log("Deployer balance:", ethers.formatEther(balance), "ETH\n");

  // Deploy FLPGAudit
  console.log("Deploying FLPGAudit contract...");
  const FLPGAudit = await ethers.getContractFactory("FLPGAudit");
  const audit = await FLPGAudit.deploy();
  await audit.waitForDeployment();

  const auditAddress = await audit.getAddress();
  console.log("✅ FLPGAudit deployed to:", auditAddress);

  let oracleAddress = deployer.address;
  if (process.env.ORACLE_PRIVATE_KEY) {
    oracleAddress = new ethers.Wallet(process.env.ORACLE_PRIVATE_KEY).address;
  } else if (process.env.ORACLE_ADDRESS) {
    oracleAddress = process.env.ORACLE_ADDRESS;
  }

  if (oracleAddress.toLowerCase() !== deployer.address.toLowerCase()) {
    console.log("Updating contract oracle to:", oracleAddress);
    const setOracleTx = await audit.setOracle(oracleAddress);
    await setOracleTx.wait();
  }

  // Verify deployment
  const owner = await audit.owner();
  const oracle = await audit.oracle();
  console.log("\nContract owner:", owner);
  console.log("Contract oracle:", oracle);

  console.log("\n========================================");
  console.log("🎉 Deployment Complete!");
  console.log("========================================");
  console.log("\n📝 Add this to your .env file:");
  console.log(`L4_CONTRACT_ADDRESS=${auditAddress}`);
  console.log(`L4_ORACLE_ADDRESS=${oracle}`);
  console.log("\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
