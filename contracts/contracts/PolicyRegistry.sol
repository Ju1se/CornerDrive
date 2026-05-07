// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title PolicyRegistry
 * @notice Registry for FLPG policy hash commitments
 *
 * This contract provides an on-chain record of policy hashes
 * for auditability and transparency. It does NOT store full
 * policies or execute policy logic - only references.
 *
 * The policy evaluation remains off-chain to preserve
 * determinism and avoid gas costs.
 */
contract PolicyRegistry {
    /// @notice Policy committed event
    event PolicyCommitted(
        uint256 indexed roundId,
        bytes32 policyHash,
        address indexed committer,
        uint256 timestamp
    );

    /// @notice Policy updated event (when a policy is modified)
    event PolicyUpdated(
        uint256 indexed roundId,
        bytes32 oldPolicyHash,
        bytes32 newPolicyHash,
        address indexed updater
    );

    /// @notice Oracle address authorized to commit policies
    address public policyOracle;

    /// @notice Policy hash for each round
    mapping(uint256 => bytes32) public policyHashes;

    /// @notice Commit timestamp for each round
    mapping(uint256 => uint256) public commitTimestamps;

    /// @notice Committer for each round
    mapping(uint256 => address) public committers;

    /// @notice Current round number
    uint256 public currentRound;

    /**
     * @notice Constructor
     * @param _oracle Initial policy oracle address
     */
    constructor(address _oracle) {
        policyOracle = _oracle;
        currentRound = 0;
    }

    /**
     * @notice Commit a policy hash for a round
     * @param roundId Round number
     * @param policyHash SHA256 hash of policy parameters
     */
    function commitPolicy(
        uint256 roundId,
        bytes32 policyHash
    ) external onlyOracle {
        bytes32 oldHash = policyHashes[roundId];

        policyHashes[roundId] = policyHash;
        commitTimestamps[roundId] = block.timestamp;
        committers[roundId] = msg.sender;

        if (roundId >= currentRound) {
            currentRound = roundId + 1;
        }

        if (oldHash == bytes32(0)) {
            emit PolicyCommitted(roundId, policyHash, msg.sender, block.timestamp);
        } else {
            emit PolicyUpdated(roundId, oldHash, policyHash, msg.sender);
        }
    }

    /**
     * @notice Batch commit multiple policy hashes
     * @param roundIds Array of round numbers
     * @param hashes Array of policy hashes
     */
    function commitPolicyBatch(
        uint256[] calldata roundIds,
        bytes32[] calldata hashes
    ) external onlyOracle {
        require(
            roundIds.length == hashes.length,
            "Arrays must have same length"
        );

        for (uint256 i = 0; i < roundIds.length; i++) {
            bytes32 oldHash = policyHashes[roundIds[i]];

            policyHashes[roundIds[i]] = hashes[i];
            commitTimestamps[roundIds[i]] = block.timestamp;
            committers[roundIds[i]] = msg.sender;

            if (roundIds[i] >= currentRound) {
                currentRound = roundIds[i] + 1;
            }

            if (oldHash == bytes32(0)) {
                emit PolicyCommitted(
                    roundIds[i],
                    hashes[i],
                    msg.sender,
                    block.timestamp
                );
            } else {
                emit PolicyUpdated(
                    roundIds[i],
                    oldHash,
                    hashes[i],
                    msg.sender
                );
            }
        }
    }

    /**
     * @notice Get policy hash for a round
     * @param roundId Round number
     * @return Policy hash (0 if not found)
     */
    function getPolicyHash(uint256 roundId) external view returns (bytes32) {
        return policyHashes[roundId];
    }

    /**
     * @notice Verify a policy hash
     * @param roundId Round number
     * @param expectedHash Expected policy hash
     * @return True if hash matches
     */
    function verifyPolicyHash(
        uint256 roundId,
        bytes32 expectedHash
    ) external view returns (bool) {
        return policyHashes[roundId] == expectedHash;
    }

    /**
     * @notice Get full policy commitment data
     * @param roundId Round number
     * @return hash Policy hash
     * @return timestamp Commit timestamp
     * @return committer Address that committed
     */
    function getPolicyCommitment(
        uint256 roundId
    ) external view returns (
        bytes32 hash,
        uint256 timestamp,
        address committer
    ) {
        return (
            policyHashes[roundId],
            commitTimestamps[roundId],
            committers[roundId]
        );
    }

    /**
     * @notice Change the policy oracle
     * @param newOracle New oracle address
     */
    function setPolicyOracle(address newOracle) external onlyOracle {
        policyOracle = newOracle;
    }

    /**
     * @notice Modifier to restrict access to oracle
     */
    modifier onlyOracle() {
        require(msg.sender == policyOracle, "Not authorized oracle");
        _;
    }
}
