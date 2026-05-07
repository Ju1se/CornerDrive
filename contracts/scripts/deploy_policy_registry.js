const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  const networkName = hre.network.name;

  console.log("Deploying PolicyRegistry with account:", deployer.address);
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("Account balance:", balance.toString());

  // Get oracle address from environment or use deployer
  let oracleAddress = deployer.address;
  if (process.env.POLICY_ORACLE_PRIVATE_KEY) {
    oracleAddress = new hre.ethers.Wallet(process.env.POLICY_ORACLE_PRIVATE_KEY).address;
  } else if (process.env.POLICY_ORACLE_ADDRESS) {
    oracleAddress = process.env.POLICY_ORACLE_ADDRESS;
  }

  // Deploy contract
  const PolicyRegistry = await hre.ethers.getContractFactory("PolicyRegistry");
  const policyRegistry = await PolicyRegistry.deploy(oracleAddress);
  await policyRegistry.waitForDeployment();

  const policyRegistryAddress = await policyRegistry.getAddress();
  console.log("PolicyRegistry deployed to:", policyRegistryAddress);
  console.log("Policy oracle set to:", oracleAddress);

  // Wait for a few block confirmations
  if (networkName !== "hardhat" && networkName !== "localhost") {
    console.log("Waiting for block confirmations...");
    const deploymentTx = policyRegistry.deploymentTransaction();
    if (deploymentTx) {
      await deploymentTx.wait(5);
    }
    console.log("Confirmed!");
  }

  // Verify contract (if on testnet/mainnet and API key is available)
  if (
    networkName !== "hardhat" &&
    networkName !== "localhost" &&
    process.env.ETHERSCAN_API_KEY
  ) {
    console.log("Verifying contract on Etherscan...");
    try {
      await hre.run("verify:verify", {
        address: policyRegistryAddress,
        constructorArguments: [oracleAddress],
      });
      console.log("Contract verified!");
    } catch (error) {
      console.log("Verification failed:", error.message);
    }
  }

  // Export deployment info
  const deploymentInfo = {
    network: networkName,
    contract: "PolicyRegistry",
    address: policyRegistryAddress,
    oracle: oracleAddress,
    deployer: deployer.address,
    deploymentDate: new Date().toISOString(),
  };

  console.log("\nDeployment info:");
  console.log(JSON.stringify(deploymentInfo, null, 2));
  console.log(`POLICY_REGISTRY_ADDRESS=${policyRegistryAddress}`);

  return policyRegistryAddress;
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
