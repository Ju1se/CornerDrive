// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title FLPG Audit Logic Contract
 * @notice Simplified L4 settlement with SBT credit system
 * @dev Enforces E[U_honest] > E[U_malicious] < 0
 */
contract FLPGAudit {
    uint256 public constant BASIS_POINTS = 10_000;
    uint256 public constant BASE_FRAUD_SLASH_BPS = 5_000;

    // ============ STATE ============
    address public owner;
    address public oracle;

    // SBT Credit System
    mapping(address => int256) public reputation;
    mapping(address => uint256) public contributions;
    mapping(address => uint256) public fraudCount;
    mapping(address => uint256) public rarityCount;
    mapping(address => uint256) public stakedAmount;
    mapping(bytes32 => bool) public processedSettlementIds;
    mapping(address => uint256) private honestRewardCarryBps;

    struct RoundEconomicPolicy {
        uint256 honestRewardMultiplierBps;
        uint256 slashMultiplierBps;
        uint256 rarityRewardMultiplierBps;
        bool configured;
        bool settlementLocked;
    }

    mapping(uint256 => RoundEconomicPolicy) private roundPolicies;

    // Constants
    int256 public constant HONEST_REWARD = 1;
    int256 public constant RARITY_REWARD = 10;
    int256 public constant FRAUD_PENALTY = -50;
    uint256 public minimumStake = 0.01 ether;

    // Tier thresholds
    int256 public constant TIER_SILVER = 100;
    int256 public constant TIER_GOLD = 500;
    int256 public constant TIER_PLATINUM = 1000;

    // ============ EVENTS ============
    event VehicleRegistered(address indexed vehicle, uint256 stake);
    event HonestContribution(address indexed vehicle, int256 newReputation);
    event RarityDiscovered(address indexed vehicle, int256 newReputation);
    event FraudDetected(address indexed vehicle, int256 newReputation);
    event StakeSlashed(address indexed vehicle, uint256 amount);
    event RoundEconomicPolicySet(
        uint256 indexed roundId,
        uint256 honestRewardMultiplierBps,
        uint256 slashMultiplierBps,
        uint256 rarityRewardMultiplierBps
    );
    event BatchSettled(
        uint256 indexed roundId,
        bytes32 indexed settlementId,
        uint256 honestCount,
        uint256 rarityCount,
        uint256 fraudCount
    );

    // ============ MODIFIERS ============
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier onlyOracle() {
        require(msg.sender == oracle, "Only oracle");
        _;
    }

    // ============ CONSTRUCTOR ============
    constructor() {
        owner = msg.sender;
        oracle = msg.sender;  // Initially owner is oracle
    }

    // ============ REGISTRATION ============
    function registerVehicle() external payable {
        require(msg.value >= minimumStake, "Insufficient stake");
        require(stakedAmount[msg.sender] == 0, "Already registered");

        stakedAmount[msg.sender] = msg.value;
        reputation[msg.sender] = 0;

        emit VehicleRegistered(msg.sender, msg.value);
    }

    // ============ SETTLEMENT ============
    function recordHonest(address vehicle) external onlyOracle {
        _recordHonest(vehicle, BASIS_POINTS);
    }

    function recordRarity(address vehicle) external onlyOracle {
        _recordRarity(vehicle, BASIS_POINTS);
    }

    function recordFraud(address vehicle) external onlyOracle {
        _recordFraud(vehicle, BASIS_POINTS);
    }

    function setRoundEconomicPolicy(
        uint256 roundId,
        uint256 honestRewardMultiplierBps,
        uint256 slashMultiplierBps,
        uint256 rarityRewardMultiplierBps
    ) external onlyOracle {
        require(
            honestRewardMultiplierBps >= 8_000 && honestRewardMultiplierBps <= 12_000,
            "Invalid honest reward multiplier"
        );
        require(
            slashMultiplierBps >= 5_000 && slashMultiplierBps <= 20_000,
            "Invalid slash multiplier"
        );
        require(
            rarityRewardMultiplierBps >= 5_000 && rarityRewardMultiplierBps <= 20_000,
            "Invalid rarity multiplier"
        );
        require(!roundPolicies[roundId].settlementLocked, "Round locked");

        roundPolicies[roundId] = RoundEconomicPolicy({
            honestRewardMultiplierBps: honestRewardMultiplierBps,
            slashMultiplierBps: slashMultiplierBps,
            rarityRewardMultiplierBps: rarityRewardMultiplierBps,
            configured: true,
            settlementLocked: false
        });

        emit RoundEconomicPolicySet(
            roundId,
            honestRewardMultiplierBps,
            slashMultiplierBps,
            rarityRewardMultiplierBps
        );
    }

    function settleBatch(
        uint256 roundId,
        address[] calldata honestVehicles,
        address[] calldata rarityVehicles,
        address[] calldata fraudVehicles,
        bytes32 settlementId
    ) external onlyOracle {
        require(settlementId != bytes32(0), "Invalid settlement id");
        require(!processedSettlementIds[settlementId], "Settlement already processed");

        RoundEconomicPolicy storage roundPolicy = roundPolicies[roundId];
        require(roundPolicy.configured, "Round policy not configured");

        processedSettlementIds[settlementId] = true;
        roundPolicy.settlementLocked = true;

        for (uint256 i = 0; i < honestVehicles.length; i++) {
            _recordHonest(honestVehicles[i], roundPolicy.honestRewardMultiplierBps);
        }

        for (uint256 i = 0; i < rarityVehicles.length; i++) {
            _recordRarity(rarityVehicles[i], roundPolicy.rarityRewardMultiplierBps);
        }

        for (uint256 i = 0; i < fraudVehicles.length; i++) {
            _recordFraud(fraudVehicles[i], roundPolicy.slashMultiplierBps);
        }

        emit BatchSettled(
            roundId,
            settlementId,
            honestVehicles.length,
            rarityVehicles.length,
            fraudVehicles.length
        );
    }

    function getRoundEconomicPolicy(uint256 roundId) external view returns (
        uint256 honestRewardMultiplierBps,
        uint256 slashMultiplierBps,
        uint256 rarityRewardMultiplierBps,
        bool configured,
        bool settlementLocked
    ) {
        RoundEconomicPolicy memory roundPolicy = roundPolicies[roundId];
        return (
            roundPolicy.honestRewardMultiplierBps,
            roundPolicy.slashMultiplierBps,
            roundPolicy.rarityRewardMultiplierBps,
            roundPolicy.configured,
            roundPolicy.settlementLocked
        );
    }

    // ============ VIEW FUNCTIONS ============
    function getTier(address vehicle) public view returns (string memory) {
        int256 rep = reputation[vehicle];
        if (rep >= TIER_PLATINUM) return "PLATINUM";
        if (rep >= TIER_GOLD) return "GOLD";
        if (rep >= TIER_SILVER) return "SILVER";
        return "BRONZE";
    }

    function getTierMultiplier(address vehicle) public view returns (uint256) {
        int256 rep = reputation[vehicle];
        if (rep >= TIER_PLATINUM) return 200;  // 2.0x
        if (rep >= TIER_GOLD) return 150;      // 1.5x
        if (rep >= TIER_SILVER) return 120;    // 1.2x
        return 100;                             // 1.0x
    }

    function getVehicleStats(address vehicle) external view returns (
        int256 rep,
        uint256 contribs,
        uint256 frauds,
        uint256 rarities,
        uint256 stake,
        string memory tier
    ) {
        return (
            reputation[vehicle],
            contributions[vehicle],
            fraudCount[vehicle],
            rarityCount[vehicle],
            stakedAmount[vehicle],
            getTier(vehicle)
        );
    }

    function isRegistered(address vehicle) external view returns (bool) {
        return stakedAmount[vehicle] > 0;
    }

    // ============ ADMIN ============
    function setOracle(address _oracle) external onlyOwner {
        oracle = _oracle;
    }

    function setMinimumStake(uint256 _stake) external onlyOwner {
        minimumStake = _stake;
    }

    function _recordHonest(address vehicle, uint256 honestRewardMultiplierBps) internal {
        require(stakedAmount[vehicle] > 0, "Not registered");

        // Preserve fractional honest incentives without changing the public
        // reputation unit: carry basis-point residue forward per vehicle.
        uint256 accumulatedRewardBps =
            honestRewardCarryBps[vehicle] +
            (uint256(HONEST_REWARD) * honestRewardMultiplierBps);
        uint256 rewardPoints = accumulatedRewardBps / BASIS_POINTS;
        honestRewardCarryBps[vehicle] = accumulatedRewardBps % BASIS_POINTS;

        reputation[vehicle] += int256(rewardPoints);
        contributions[vehicle]++;

        emit HonestContribution(vehicle, reputation[vehicle]);
    }

    function _recordRarity(address vehicle, uint256 rarityMultiplierBps) internal {
        require(stakedAmount[vehicle] > 0, "Not registered");

        int256 dynamicReward = (RARITY_REWARD * int256(rarityMultiplierBps)) / int256(BASIS_POINTS);
        reputation[vehicle] += dynamicReward;
        contributions[vehicle]++;
        rarityCount[vehicle]++;

        emit RarityDiscovered(vehicle, reputation[vehicle]);
    }

    function _recordFraud(address vehicle, uint256 slashMultiplierBps) internal {
        require(stakedAmount[vehicle] > 0, "Not registered");

        uint256 penaltyMagnitude = (50 * slashMultiplierBps) / BASIS_POINTS;
        reputation[vehicle] -= int256(penaltyMagnitude);
        fraudCount[vehicle]++;

        // Keep the previous 50% slash as the 1.0x baseline, capped at a full slash.
        uint256 slashRateBps = (BASE_FRAUD_SLASH_BPS * slashMultiplierBps) / BASIS_POINTS;
        if (slashRateBps > BASIS_POINTS) {
            slashRateBps = BASIS_POINTS;
        }

        uint256 slashAmount = (stakedAmount[vehicle] * slashRateBps) / BASIS_POINTS;
        stakedAmount[vehicle] -= slashAmount;

        payable(owner).transfer(slashAmount);

        emit FraudDetected(vehicle, reputation[vehicle]);
        emit StakeSlashed(vehicle, slashAmount);
    }

    // Receive ETH
    receive() external payable {}
}
