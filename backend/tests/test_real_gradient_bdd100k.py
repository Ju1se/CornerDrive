import json

import pytest

from policy_agent.analysis.real_gradient_benchmark import (
    RealGradientBenchmarkConfig,
    build_real_gradient_data_bundle,
    load_bdd100k_clients,
    load_leaf_femnist_clients,
    load_real_clients,
    make_real_data_adaptive_policy,
    run_real_gradient_benchmark,
)


def _write_image(path, color):
    image_module = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image_module.new("RGB", (12, 8), color=color)
    image.save(path)


def _make_bdd_fixture(root):
    image_root = root / "images" / "100k" / "train"
    records = []
    specs = [
        ("clear_day_1.jpg", "clear", "daytime", "city street", (220, 220, 220)),
        ("clear_day_2.jpg", "clear", "daytime", "city street", (200, 200, 200)),
        ("rain_night_1.jpg", "rainy", "night", "highway", (30, 30, 80)),
        ("rain_night_2.jpg", "rainy", "night", "highway", (35, 35, 90)),
        ("snow_day_1.jpg", "snowy", "daytime", "residential", (240, 240, 255)),
        ("snow_day_2.jpg", "snowy", "daytime", "residential", (245, 245, 255)),
    ]
    for name, weather, timeofday, scene, color in specs:
        _write_image(image_root / name, color)
        records.append(
            {
                "name": name,
                "attributes": {
                    "weather": weather,
                    "timeofday": timeofday,
                    "scene": scene,
                },
                "labels": [],
            }
        )

    label_file = root / "labels" / "bdd100k_labels_images_train.json"
    label_file.parent.mkdir(parents=True, exist_ok=True)
    label_file.write_text(json.dumps(records), encoding="utf-8")
    return label_file, image_root


def _write_leaf_shard(path, *, users, samples_per_user):
    payload = {"users": list(users), "num_samples": [], "user_data": {}}
    for user_index, user in enumerate(users):
        sample_count = samples_per_user[user_index]
        payload["num_samples"].append(sample_count)
        payload["user_data"][user] = {
            "x": [
                [float((sample_index + feature_index + user_index) % 2) for feature_index in range(784)]
                for sample_index in range(sample_count)
            ],
            "y": [
                (user_index + sample_index) % 10
                for sample_index in range(sample_count)
            ],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_leaf_fixture(root):
    train_file = root / "train" / "all_data_0_niid_05_keep_0_train_9.json"
    test_file = root / "test" / "all_data_0_niid_05_keep_0_test_9.json"
    _write_leaf_shard(train_file, users=["train_a", "train_b", "train_c"], samples_per_user=[8, 8, 8])
    _write_leaf_shard(test_file, users=["test_a", "test_b", "test_c", "test_d"], samples_per_user=[8, 8, 8, 8])
    return train_file, test_file


def test_load_bdd100k_clients_groups_attribute_pseudo_clients(tmp_path):
    label_file, image_root = _make_bdd_fixture(tmp_path)

    clients, info = load_bdd100k_clients(
        tmp_path,
        label_file=str(label_file),
        image_dir=str(image_root),
        image_size=8,
        target_attribute="weather",
        client_group="weather_timeofday",
        corner_values="rainy,snowy",
        max_clients=10,
        min_samples_per_client=2,
        max_samples_per_client=2,
        seed=7,
    )

    assert info["source"] == "bdd100k"
    assert info["target_attribute"] == "weather"
    assert info["real_client_partitions"] is False
    assert set(info["class_to_id"]) == {"clear", "rainy", "snowy"}
    assert {
        info["id_to_class"][label]
        for label in info["corner_labels"]
    } == {"rainy", "snowy"}
    assert len(clients) == 3
    assert all(client.inputs.shape == (2, 8 * 8 * 3) for client in clients)
    assert all(client.targets.shape == (2,) for client in clients)
    assert all(client.metadata["client_group"] == "weather_timeofday" for client in clients)


def test_load_real_clients_supports_bdd100k_source(tmp_path):
    label_file, image_root = _make_bdd_fixture(tmp_path)
    config = RealGradientBenchmarkConfig(
        source="bdd100k",
        bdd_data_dir=str(tmp_path),
        bdd_label_file=str(label_file),
        bdd_image_dir=str(image_root),
        bdd_image_size=8,
        bdd_corner_values="rainy,snowy",
        max_clients=2,
        min_samples_per_client=2,
        max_samples_per_client=2,
    )

    clients, info = load_real_clients(config)

    assert len(clients) == 2
    assert info["source"] == "bdd100k"
    assert info["corner_labels"]


def test_leaf_loader_defaults_to_train_split(tmp_path):
    _make_leaf_fixture(tmp_path)

    clients, info = load_leaf_femnist_clients(
        tmp_path,
        max_clients=10,
        min_samples_per_client=2,
        max_samples_per_client=8,
    )

    assert info["split"] == "train"
    assert {client.client_id for client in clients} == {"train_a", "train_b", "train_c"}


def test_real_gradient_bundle_uses_leaf_test_for_heldout_eval(tmp_path):
    _make_leaf_fixture(tmp_path)
    config = RealGradientBenchmarkConfig(
        source="femnist",
        leaf_data_dir=str(tmp_path),
        max_clients=3,
        min_samples_per_client=2,
        max_samples_per_client=8,
        reference_split_fraction=0.5,
    )

    bundle = build_real_gradient_data_bundle(config)

    assert {client.client_id for client in bundle.clients} == {"train_a", "train_b", "train_c"}
    assert bundle.dataset_info["split_protocol"] == "leaf_train_clients_test_reference_eval"
    assert bundle.dataset_info["update_split"] == "leaf_train"
    assert bundle.dataset_info["reference_split"] == "leaf_test_reference_clients"
    assert bundle.dataset_info["evaluation_split"] == "leaf_test_evaluation_clients"
    assert bundle.dataset_info["audit_main_sample_count"] > 0
    assert bundle.dataset_info["eval_main_sample_count"] > 0


def test_bdd100k_source_runs_real_gradient_smoke(tmp_path):
    label_file, image_root = _make_bdd_fixture(tmp_path)
    config = RealGradientBenchmarkConfig(
        source="bdd100k",
        bdd_data_dir=str(tmp_path),
        bdd_label_file=str(label_file),
        bdd_image_dir=str(image_root),
        bdd_image_size=8,
        bdd_corner_values="rainy,snowy",
        max_clients=3,
        min_samples_per_client=2,
        max_samples_per_client=2,
        clients_per_round=3,
        rounds=1,
        pretrain_steps=1,
        local_batch_size=2,
        attack_fraction=0.0,
        corner_harm_fraction=0.0,
        noise_fraction=0.0,
    )

    result = run_real_gradient_benchmark(config)

    assert result["dataset"]["source"] == "bdd100k"
    assert result["dataset"]["input_dim"] == 8 * 8 * 3
    for method_id in ("krum", "fltrust", "zeno", "zenopp", "cornerdrive"):
        assert result["methods"][method_id]["round_records"]


def test_real_data_adaptive_profile_routes_cornerdrive_l1v3(tmp_path):
    label_file, image_root = _make_bdd_fixture(tmp_path)
    config = RealGradientBenchmarkConfig(
        source="bdd100k",
        bdd_data_dir=str(tmp_path),
        bdd_label_file=str(label_file),
        bdd_image_dir=str(image_root),
        bdd_image_size=8,
        bdd_corner_values="rainy,snowy",
        max_clients=3,
        min_samples_per_client=2,
        max_samples_per_client=2,
        clients_per_round=3,
        rounds=1,
        pretrain_steps=1,
        local_batch_size=2,
        attack_fraction=0.0,
        corner_harm_fraction=0.0,
        noise_fraction=0.0,
        cornerdrive_l1_mode="v3_m2_norm_sign_fixed",
        cornerdrive_l1_norm_mad_threshold=2.5,
        cornerdrive_l1_sign_threshold=0.55,
    )

    result = run_real_gradient_benchmark(config, policy=make_real_data_adaptive_policy())
    round_record = result["methods"]["cornerdrive"]["round_records"][0]

    assert result["policy"]["theta_tol"] == 0.02
    assert round_record["l1_router_mode"] == "v3_m2_norm_sign_fixed"
    assert "l1_review_rate" in round_record
